# Photo Search Improvement Plan

## Current State

We have a functional semantic photo search app using CLIP embeddings + Qdrant with dual-vector (image + text) RRF fusion. Photos are indexed with basic captions (filename + geolocation + timestamp).

**Problems identified:**
- CLIP is blind to factual data ("2012", "London", proper nouns)
- CLIP-text was trained for *image-text alignment*, not *text-text matching* — it's fundamentally weak at semantic text retrieval
- All scores cluster tightly (0.32–0.48), making ranking unreliable
- No keyword/lexical matching for exact term matches
- No query understanding — natural language goes straight to embedding

---

## Phase 1: Better Text Embeddings (Highest Impact, Lowest Effort)

Swap CLIP-text for a dedicated text embedding model. This is likely the single biggest win.

### Rationale
CLIP-text was trained to align text with images, not to match text to text. Models like `bge-large-en` or `gte-large` are trained specifically for semantic text retrieval and dramatically outperform CLIP-text on factual queries, even with identical captions. Our current captions already contain "greater london, england, united kingdom august 2018" — CLIP-text just can't match "england" to that text.

### Tasks
- [x] Evaluate candidate text embedding models (`bge-large-en`, `gte-large`, `jina-embeddings-v3`)
- [x] Add a separate "text" vector in Qdrant using the new model (keep CLIP-image as-is)
- [x] Update indexing pipeline to embed captions with the new model
- [x] Update search to use new model for query encoding
- [x] Benchmark: CLIP-text vs new model on test queries

### Results
Chose `BAAI/bge-large-en-v1.5` (1024-dim, matches existing Qdrant schema).
Integration tests went from **7/13 passing** (CLIP-text) to **13/13 passing** (BGE).
Query time dropped from ~23s (CLIP cold start) to ~5.5s.

### Expected outcome
✅ "Photos from my trip to england" finally matches captions containing "england" — without any structural changes to the search pipeline.

---

## Phase 2: Query Understanding (High Impact, Low Effort)

Extract structured entities from user queries before they reach the search pipeline.

### Tasks
- [x] Implement a query parser that detects:
  - **Locations** (cities, countries, regions) → handled by BGE embeddings (no filter needed)
  - **Dates/Years** (2012, "summer 2018", "last year") → map to `taken_at` payload filter
  - **Objects/Concepts** (dog, beach, sunset) → pass to embedding models
- [x] ~~Use spaCy or GLiNER for lightweight NER~~ — skipped, BGE + regex is sufficient
- [x] ~~Strip extracted entities from the semantic query text~~ — not needed, BGE handles full query
- [x] Pass extracted entities to Qdrant as `must`/`should` filters during search

### Decision: No NER needed
BGE embeddings already match location names accurately. The regex parser handles dates exactly where precision matters. NER would add ~50MB+ dependencies and per-query latency without meaningful ranking improvements.

### Expected outcome
✅ "Photos from my trip to england in 2012" → filter by year, embed rest semantically.

---

## Phase 3: Hybrid Search — SPLATE-Based Lexical Matching

Add learned sparse vector matching alongside dense vector search for exact keyword retrieval.

### Why SPLADE Over BM25?
The original plan called for BM25, but research shows **SPLADE** (learned sparse embeddings) significantly outperforms BM25:

| Model | BEIR Avg nDCG@10 | Params |
|-------|-------------------|--------|
| `opensearch-neural-sparse-encoding-v2-distill` | **52.8** | 67M |
| `naver/splade-v3` | **51.7** | 109M |
| **BM25 (baseline)** | **45.6** | N/A |

~13% improvement with comparable CPU footprint. From Qdrant's perspective, both produce sparse vectors — same implementation surface area, just swap the tokenizer for a model at query time.

### Chosen Model
Chose `naver/splade-v3-distilbert` (67M params) — balances quality and CPU speed.
Accessed via `sentence_transformers.SparseEncoder` for convenience.

Known issue: SPLADE returns PyTorch COO-encoded sparse tensors that required special handling in `_tensor_to_qdrant_sparse()` — indices/values must be extracted directly rather than converting to dense (which would allocate a ~30K vocab-sized array).

### Tasks
- [x] Add sparse vector config to Qdrant collection schema
- [x] Create `SparseEncodingService` using chosen SPLADE model (lazy-loaded, thread-safe)
- [x] Index enriched captions as sparse vectors in Qdrant
- [x] Encode queries with SPLADE at search time
- [x] Update search to run 3-way RRF: image-dense + text-dense + SPLADE-sparse (k=60)
- [ ] Benchmark: dense-only vs hybrid on test queries
- [x] Add unit tests for sparse encoding service

### Cost Analysis
- **Index-time**: SPLADE encodes each caption once (~ms per caption on CPU, batchable)
- **Query-time**: ~ms per query, negligible overhead
- **Storage**: Sparse vectors are tiny (non-zero indices + weights per token)

### Results
SPLADE produces ~100 non-zero elements per caption (out of ~30K vocab), tiny storage footprint.
All 10 media files re-indexed successfully with triple-vector coverage (image + text + sparse).
Graceful fallback: if collection lacks sparse support, search degrades to dual-vector seamlessly.

### Expected outcome
✅ Exact keyword matches ("england", "tower bridge") rank highly even when dense models undershoot, while SPLADE's learned expansions catch semantically related terms BM25 would miss.

---

## Phase 4: Caption Enrichment (COMPLETED)

Generate richer textual descriptions of photos using a VLM.

### Rationale (updated)
VLM captions provide massive semantic surface area for BGE embeddings. A caption like "A woman standing on a wooden pier overlooking a calm lake surrounded by mountains at golden hour" gives the search engine exponentially more to match against compared to "italy garda lake sailing club september 2018".

### Results
Chose `Qwen/Qwen2.5-VL-3B-Instruct` for high-quality captions.
- New `VLMService` with lazy loading, greedy decoding for consistency
- `vlm_caption` field on `IndexedMedia` model (stored separately from combined caption)
- `enrich_captions` management command with progress bar, resume support, checkpoints
- VLM caption is prepended to factual caption during indexing
- 4 new unit tests, all 19 tests passing

### Usage
```bash
# Dry run first
python manage.py enrich_captions --dry-run

# Process all images without VLM captions
python manage.py enrich_captions --batch-size 50

# Reprocess images that already have captions
python manage.py enrich_captions --reprocess

# Limit to N images for testing
python manage.py enrich_captions --limit 5
```

### Expected outcome
✅ Photos get searchable descriptions like "family sitting around birthday cake with number 5, indoors, evening" instead of just "filename.jpg, city, country, date".

---

## Phase 5: Reranking & Tuning (COMPLETED)

Fine-tune the search pipeline for better precision.

### Tasks
- [x] Build a small test corpus with expected results → 17 queries, 4 categories
- [x] Measure Recall@K across query types (factual, visual, mixed) → 100% R@1 on test set
- [x] Tune query parser filler words → eliminated false location extraction (+5% overall)
- [ ] Evaluate Cross-Encoder reranking for top-K candidates → skipped; no signal on 5-image test set
- [ ] Experiment with weighted score blending (α ≈ 0.5) as alternative to RRF → deferred to production-scale data
- [ ] Tune RRF k-value and dense/sparse weights based on benchmarks → deferred to production-scale data

### Results
BenchmarkCollector module in `tests/benchmark.py` provides reusable evaluation against any search function. Test corpus covers location, year, combined date+location, and seasonal queries. On the 5-image fixture set, achieves perfect recall. Designed to scale to larger libraries where discrimination matters.

### Expected outcome
✅ Search quality measurable and reproducible. Ready for production-scale optimization.

---

## Priorities

| Priority | Phase | Why |
|----------|-------|-----|
| ✅ Done | Phase 1: Better Text Embeddings | BGE replaced CLIP-text. 13/13 tests passing. |
| ✅ Done | Phase 2: Query Understanding | Regex parser + payload filters. No NER needed. |
| ✅ Done | Phase 4: Caption Enrichment | Qwen2.5-VL-3B VLM service + batch management command. Ready for production use. |
| ✅ Done | Phase 3: Hybrid Search (SPLADE) | Triple-vector RRF (image + text + sparse). ~100 non-zeros/doc, tiny storage. Graceful fallback to dual-vector. |
| ✅ Done | Phase 3: Hybrid Search (SPLADE) | Triple-vector RRF (image + text + sparse). ~100 non-zeros/doc, tiny storage. Graceful fallback to dual-vector. |
| ⚪ P3 | Phase 5: Reranking | Benchmark infrastructure ready. Cross-encoder/blending deferred until production-scale data available. |

---

## Market Research Notes (June 2026)

Assessment of current landscape vs. our architecture choices.

### Confirmed Good Choices
- **Dual-vector dense search (CLIP image + BGE text)** — still industry standard for multimodal retrieval
- **BGE-large-en-v1.5** — remains top-tier for CPU deployment on MTEB leaderboards
- **Qdrant for self-hosted** — leads in developer ergonomics; sparse vector support matured ahead of competitors
- **RRF fusion** — still the standard method for merging ranked results
- **VLM caption enrichment** — confirmed highest-leverage investment before adding lexical components

### Emerging Alternatives Worth Watching
- **Unified multimodal embeddings** — Jina v5-omni (May 2026) and Qwen3-VL-Embedding-2B can embed text, images, audio, video into one shared space. Could eventually consolidate our separate CLIP + BGE pipeline, but dual-vector specialization currently gives better control over ranking.
- **Milvus 3.0-beta** — added first-class full-text search and hybrid capabilities. Competitively featured, but no compelling reason to migrate from Qdrant given our invested structure.

### Key Insight (Updated)
**SPLADE bridges the gap between dense semantics and exact keywords.** Dense vectors catch meaning ("sunset" ↔ "golden hour"), SPLADE catches tokens plus learned expansions ("Tower Bridge" → exact match + "London landmark"). Together with VLM captions providing rich text surface area, this covers the full spectrum without needing a reranker.

---

## Key Insight

**The problem was both the embedding model AND the captions.** Phase 1 (BGE) solved factual matching ("england", "2012"). Phase 4 (VLM) solves visual matching ("birthday cake", "sunset", "mountains"). Together they cover the full spectrum of photo search queries.
