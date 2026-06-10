# LatentSearch — Project Instructions

## Project Overview

LatentSearch is a semantic, privacy-first, context-aware indexing and search engine for self-hosted media libraries. It computes visual embeddings locally using CLIP, generates captions with a VLM model, and stores vectors in Qdrant for semantic search. Served via Django with a vanilla HTML/CSS/JS frontend.

**Repository:** `coenvdgrinten/latent-search` on GitHub
**Branch:** `main`

---

## Tech Stack

- **Python 3.13+** managed by **uv** (never use pip directly)
- **Django 6.0.6** for backend, admin, and templating
- **PyTorch** + **Hugging Face Transformers 5.x** for ML inference
- **CLIP**: `jinaai/jina-clip-v2` via `AutoModel` (1024-dim embeddings, `trust_remote_code=True`)
- **Text Embeddings**: `BAAI/bge-large-en-v1.5` via `SentenceTransformer` (1024-dim)
- **VLM Captions**: `Qwen/Qwen2.5-VL-3B-Instruct` via `Qwen2_5_VLForConditionalGeneration` (CPU-only, half precision)
- **Qdrant** vector database for semantic similarity search
- **SQLite** for Django ORM and local caches (geocoding)
- **HTMX** for AJAX-powered result swapping (no full page reloads)
- **Vanilla HTML/CSS/JS** frontend — no frameworks, no build step, no TypeScript in browser code

---

## Architecture

### Directory Structure

```
src/latent_search/server/
├── config/          # Django settings, urls, ASGI/WSGI
├── indexing/        # Main Django app
│   ├── models/      # Data models (IndexedMedia)
│   ├── services/    # Business logic layer (stateless services)
│   ├── views/       # HTTP handlers
│   ├── templates/   # Django templates
│   ├── static/      # CSS, JS assets
│   ├── management/  # Django management commands
│   │   └── commands/# index_media, enrich_captions, setup_qdrant
│   └── tests/       # Django TestCase modules
```

### Service Layer Pattern

All business logic lives in `services/`. Services are instantiated fresh per-operation (injected into `IndexingService` or `SearchService`). They use lazy-loading for heavy ML models:

```python
@property
def model(self) -> SomeModel:
    if self._model is None:
        with self._lock:
            if self._model is None:
                self._model = SomeModel.from_pretrained(...)
    return self._model
```

Multi-threading safety: double-checked locking with `threading.Lock`.

### Models

- `IndexedMedia` — core model representing a discovered media file
- Fields: `file_path` (unique key), `vlm_caption`, `is_indexed`, `vector_id`, `taken_at`, `latitude`/`longitude`
- Uses `typing.override` decorator on `__str__`

---

## Coding Conventions

### Python Style

- **Type hints everywhere** — use Python 3.13+ syntax (`X | Y` union, not `Union[X, Y]`)
- **`@override` decorator** on all overridden methods (Django `setUp`, `handle`, `__str__`, etc.)
- **PEP 8 compliant**, enforced by **Ruff** (line length 88)
- **Docstrings** on public classes and methods — triple-quoted, describe purpose and args
- **Logging** via `logging.getLogger(__name__)` — never use `print()`
- **Imports**: standard lib → third-party → local (grouped with blank lines)
- **String quotes**: double quotes preferred (`""`)

### Testing

- **Django `TestCase`** — inherit from `django.test.TestCase`
- **Mock heavy dependencies** (CLIP, VLM, Qdrant, network) with `unittest.mock.patch`
- **Patch at the import site** — e.g., `"latent_search.server.indexing.services.clip.AutoModel.from_pretrained"`
- **Naming**: `{Subject}Test` classes, `test_{behavior}` methods
- **`setUp`** method for shared fixtures — decorated with `@override`
- **Integration tests** live in `test_search_integration.py` — they hit real Qdrant + CLIP. Skip gracefully via `self.skipTest` when Qdrant is unavailable
- Run tests: `./manage test`

### Frontend

- **Vanilla JS only** — no TypeScript annotations in browser JavaScript (common pitfall!)
- **Dark theme** with glassmorphism CSS, animated background blobs
- **HTMX** for AJAX — swap `_results_fragment.html` on search
- **Responsive** grid layout for image cards
- **Lightbox modal** with keyboard navigation (arrow keys, Escape)

---

## Commands

Use the `./manage` wrapper script (not `python manage.py`) — it is the canonical entrypoint and what CI runs.

| Task | Command |
|------|---------|
| Install deps | `uv sync` |
| Run linter | `uv run ruff check` |
| Format | `uv run ruff format` |
| Type check | `uv run ty check` |
| Run all tests | `./manage test` |
| Run single test file | `./manage test latent_search.server.indexing.tests.test_clip` |
| Run single test method | `./manage test latent_search.server.indexing.tests.test_clip.CLIPServiceTest.test_get_image_embedding` |
| Run server | `./manage runserver` |
| Discover & index media | `./manage index_media /path/to/photos` |
| Generate VLM captions | `./manage enrich_captions` |
| Setup Qdrant collection | `./manage setup_qdrant` |

Ruff rules enabled: `E` (pycodestyle errors), `F` (pyflakes), `I` (isort), `N` (pep8-naming), `UP` (pyupgrade), `B` (bugbear). Target: Python 3.13+.

---

## ML Model Details

### CLIP (Image + Text Embeddings)
- Model: `jinaai/jina-clip-v2`
- Class: `transformers.AutoModel` (custom architecture, `trust_remote_code=True`)
- Dimensions: 1024
- Precision: float32 (cast after loading — loading with `torch_dtype=float32` breaks Jina's custom loader)
- CPU quirk: `PYTORCH_DISABLE_AVX512_BF16_MATMUL=1` env var required to prevent NaN outputs

### Text Embeddings
- Model: `BAAI/bge-large-en-v1.5`
- Library: `sentence_transformers.SentenceTransformer`
- Dimensions: 1024
- Normalization: `normalize_embeddings=True` in `encode()`
- Always use `torch.no_grad()` context for inference

### VLM (Image Captioning)
- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Class: `Qwen2_5_VLForConditionalGeneration` (NOT `Qwen2VLForConditionalGeneration`)
- Precision: `torch.float16` (half precision for memory efficiency on CPU)
- System prompt guides output to <200 char search-optimized descriptions
- Each image takes several minutes on CPU

---

## Search Pipeline

Search uses **dual-vector RRF (Reciprocal Rank Fusion)**:
1. **Image vector** — CLIP image embedding, catches visual matches (scenes, composition)
2. **Text vector** — BGE text embedding of the combined caption, catches semantic text matches

Both vectors share the same 1024-dim space. Qdrant prefetches the image-vector results, then reranks by text-vector similarity. RRF merges the two rankings automatically.

Payload filters (year, month, season, location) are applied to both stages via `ParsedQuery`.

Connection failures to Qdrant are wrapped in `QdrantUnavailableError` so the view can render a friendly error fragment.

## Views

Views use a **module-level singleton** for `SearchService` (instantiated at import time). This avoids recreating the Qdrant client on every request. Do not inject the service per-request.

HTMX detection: check `request.headers.get("HX-Request")` to decide whether to render the full dashboard or just a fragment (`_results_fragment.html` or `_error_fragment.html`).

## Management Commands

Commands follow a thin-controller pattern: argument parsing in `add_arguments`/`handle`, all business logic delegated to services. The `enrich_captions` command demonstrates the pattern for resumable batch work (state file at `~/.latent_search_vlm_state.json`, checkpoint saves, Ctrl+C handling).

---

## Common Pitfalls

1. **TypeScript in browser JS** — the editor may auto-add type annotations (`: HTMLElement`, etc.) to JavaScript. Remove them — browsers can't parse TS.
2. **Wrong VLM class name** — use `Qwen2_5_VLForConditionalGeneration` (with underscore), not `Qwen2VLForConditionalGeneration`.
3. **Duplicate template markup** — when modifying result cards, update BOTH `dashboard.html` (initial load) and `_results_fragment.html` (HTMX swap).
4. **CLIP model loading** — load default then cast to float32, don't pass `torch_dtype` to `from_pretrained`.
5. **Patch paths in tests** — always patch where the class is imported, not where it's defined.
6. **Environment variables** — read from `.env` via `python-dotenv`; settings.py calls `load_dotenv()`.
