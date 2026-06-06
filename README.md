# LatentSearch

[![Lint](https://github.com/coenvdgrinten/latent-search/actions/workflows/lint.yml/badge.svg)](https://github.com/coenvdgrinten/latent-search/actions/workflows/lint.yml)
[![Type Check](https://github.com/coenvdgrinten/latent-search/actions/workflows/typecheck.yml/badge.svg)](https://github.com/coenvdgrinten/latent-search/actions/workflows/typecheck.yml)
[![Tests](https://github.com/coenvdgrinten/latent-search/actions/workflows/test.yml/badge.svg)](https://github.com/coenvdgrinten/latent-search/actions/workflows/test.yml)
[![Verified with ty](https://img.shields.io/badge/type__checker-ty-007acc?style=flat)](https://github.com/astral-sh/ty)
[![Linted with Ruff](https://img.shields.io/badge/lint-Ruff-3670A0?style=flat&logo=python&logoColor=ffdd54)](https://github.com/astral-sh/ruff)
[![Managed by uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)

A semantic, privacy-first, context-aware indexing and search engine for self-hosted media libraries. **LatentSearch** hooks directly into your Nextcloud photo directories on Unraid, computes visual and geospatial embeddings locally using PyTorch and OpenAI's CLIP model, and stores them in a high-performance Qdrant vector database. 

The entire user experience is served via a blazing-fast, server-rendered Django application styled with minimalist modern CSS.

---

## Tech Stack & Ecosystem

* **Backend Engine:** Python 3.13 managed by **uv** for ultra-fast environment resolutions.
* **ML Layer:** PyTorch + Hugging Face Transformers (CLIP `vit-base-patch32`).
* **Vector Store:** Qdrant (Self-hosted on Unraid via Docker).
* **Web UI:** Django 5.x + Pure CSS (with native CSS Grid layout).
* **Toolchain Quality Gate:** **Ruff** for linting/formatting and **ty** for high-performance static type checking.

---

## Features

- **Local-First AI:** Privacy-aware indexing using CLIP embeddings computed on your hardware.
- **Geospatial Search:** Map-based discovery leveraging EXIF GPS data.
- **Semantic Queries:** Search for "mountains at sunset" instead of relying on filenames.
- **Nextcloud Integration:** Syncs metadata without moving your original files.

## Getting Started

### Prerequisites

- **Python 3.13+** (Managed by `uv` recommended)
- **Qdrant:** Running instance of Qdrant vector database.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/coenvdgrinten/latent-search.git
   cd latent-search
   ```

2. **Setup environment:**
   ```bash
   uv sync
   ```

3. **Configure Environment:**
   Create a `.env` file (see `.env.example` for reference):
   ```bash
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   NEXTCLOUD_PATH=/mnt/user/photos
   ```

## Development

Run quality checks locally:

```bash
uv run ruff check .
uv run ty check
```

## License

MIT
