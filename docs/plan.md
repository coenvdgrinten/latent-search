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

## Phase 3: Hybrid Search (Deferred)

Add sparse/keyword matching alongside dense vector search.

### Decision: Deferred until after Phase 4
Phase 4 (VLM captions) provides bigger search quality gains with less implementation effort. BM25 will be more valuable after VLM enrichment (more text to tokenize).

### Tasks
- [ ] Enable Qdrant sparse vectors (native support)
- [ ] Build a BM25 tokenizer for enriched captions
- [ ] Index captions as sparse vectors in Qdrant
- [ ] Update search to run dense + sparse in parallel
- [ ] Fuse results with RRF (k=60) as baseline
- [ ] Benchmark: dense-only vs sparse-only vs hybrid

### Expected outcome
Exact keyword matches ("england", "2012", "tower bridge") rank highly even when the dense model misses them.

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

## Phase 5: Reranking & Tuning (Incremental, Ongoing)

Fine-tune the search pipeline for better precision.

### Tasks
- [ ] Evaluate Cross-Encoder reranking for top-K candidates
- [ ] Experiment with weighted score blending (α ≈ 0.5) as alternative to RRF
- [ ] Build a small test corpus with expected results
- [ ] Measure Recall@K across query types (factual, visual, mixed)
- [ ] Tune RRF k-value and dense/sparse weights based on benchmarks

### Expected outcome
More consistent, measurable improvements in search quality.

---

## Priorities

| Priority | Phase | Why |
|----------|-------|-----|
| ✅ Done | Phase 1: Better Text Embeddings | BGE replaced CLIP-text. 13/13 tests passing. |
| ✅ Done | Phase 2: Query Understanding | Regex parser + payload filters. No NER needed. |
| ✅ Done | Phase 4: Caption Enrichment | Qwen2.5-VL-3B VLM service + batch management command. Ready for production use. |
| 🟢 P2 | Phase 3: Hybrid Search | Deferred — more valuable after VLM enrichment. |
| ⚪ P3 | Phase 5: Reranking | Incremental polish after foundation is solid. |

---

## Key Insight

**The problem was both the embedding model AND the captions.** Phase 1 (BGE) solved factual matching ("england", "2012"). Phase 4 (VLM) solves visual matching ("birthday cake", "sunset", "mountains"). Together they cover the full spectrum of photo search queries.
