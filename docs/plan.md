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
- [ ] Evaluate candidate text embedding models (`bge-large-en`, `gte-large`, `jina-embeddings-v3`)
- [ ] Add a separate "text" vector in Qdrant using the new model (keep CLIP-image as-is)
- [ ] Update indexing pipeline to embed captions with the new model
- [ ] Update search to use new model for query encoding
- [ ] Benchmark: CLIP-text vs new model on test queries

### Expected outcome
"Photos from my trip to england" finally matches captions containing "england" — without any structural changes to the search pipeline.

---

## Phase 2: Query Understanding (High Impact, Low Effort)

Extract structured entities from user queries before they reach the search pipeline.

### Tasks
- [ ] Implement a query parser that detects:
  - **Locations** (cities, countries, regions) → map to `caption` payload filter or BM25 boost
  - **Dates/Years** (2012, "summer 2018", "last year") → map to `taken_at` payload filter
  - **Objects/Concepts** (dog, beach, sunset) → pass to embedding models
- [ ] Use spaCy or GLiNER for lightweight NER
- [ ] Strip extracted entities from the semantic query text
- [ ] Pass extracted entities to Qdrant as `must` filters during search

### Expected outcome
"Photos from my trip to england in 2012" → filter by year/location, embed "trip" semantically.

---

## Phase 3: Hybrid Search (High Impact, Medium Effort)

Add sparse/keyword matching alongside dense vector search.

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

## Phase 4: Caption Enrichment (Lower Priority Than Initially Thought)

Generate richer textual descriptions of photos using a VLM.

### Rationale (updated)
Our current captions already include good factual data ("greater london, england, united kingdom august 2018 summer afternoon"). The problem isn't caption quality — it's that CLIP-text can't match it. After Phases 1–3, evaluate whether VLM captions still add value. They may only be worth it for expanding visual concepts (objects, scenes, actions) that aren't in the filename/metadata.

### Tasks
- [ ] Select a VLM (LLaVA, Qwen-VL, or similar) compatible with local CPU inference
- [ ] Batch-process existing library to generate descriptive captions
- [ ] Ground captions with EXIF data (inject known dates/locations into VLM prompt)
- [ ] Optionally add OCR for text inside images (street signs, menus, etc.)
- [ ] Store enriched captions in both DB and Qdrant payload
- [ ] Re-index with improved captions for both dense and sparse vectors

### Expected outcome
Photos get searchable descriptions like "family sitting around birthday cake with number 5, indoors, evening" instead of just "filename.jpg, city, country, date".

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
| 🔴 P0 | Phase 1: Better Text Embeddings | Biggest bang-for-buck — swap the model, keep everything else. May solve 80% of factual query failures alone. |
| 🟡 P1 | Phase 2: Query Understanding | Converts natural language to structured filters. Works synergistically with Phase 1. |
| 🟢 P2 | Phase 3: Hybrid Search | Catches edge cases dense model misses. Leverages Qdrant's native sparse vector support. |
| ⚪ P3 | Phase 4: Caption Enrichment | Lower priority now — depends on whether Phases 1–3 are sufficient. |
| ⚪ P4 | Phase 5: Reranking | Incremental polish after foundation is solid. |

---

## Key Insight

**The problem is likely the embedding model, not the captions.** Our captions already contain "england", "2012", "london" — CLIP-text just can't match them because it was trained for image-text alignment, not text-text retrieval. Swapping to a dedicated text embedder (Phase 1) combined with query understanding (Phase 2) may get us to production-quality results without needing VLM captions, OCR, or cross-encoders.
