# Offloading Indexing to a GPU Machine

LatentSearch is designed to run on a low-power box (e.g. an Unraid server
hosting your Nextcloud photos). VLM captioning and CLIP embedding are
expensive — a few minutes per image on CPU. This document describes how to
offload that work to a separate GPU machine, leaving the Unraid box as a
thin metadata + search server.

## How it works

```
┌──── Unraid (low power) ────┐    ┌──── GPU machine (elsewhere) ────┐
│  Nextcloud photos          │    │  ./manage offload_index          │
│  LatentSearch (Django)     │◄───┤  1. GET /api/export_library      │
│  Qdrant                    │    │     ?pending_only=true           │
│       ▲                    │    │  2. per record:                  │
│       │ import_library     │    │     GET /api/media/<id>/image    │
│       │ (write-back)       │    │  3. VLM.describe()               │
│       └────────────────────┼────┤  4. CLIP.get_image_embedding()   │
│                            │    │  5. BGE.encode(caption)          │
│                            │    │  6. VectorDB.upsert → Qdrant     │
│                            │    │  7. POST /api/import_library     │
└────────────────────────────┘    └──────────────────────────────────┘
```

The GPU machine is **stateless**: no local database, no copy of the photo
library. It pulls a record list, pulls image bytes one at a time, runs the
ML work, pushes vectors straight to the Unraid Qdrant, and pushes captions
+ `is_indexed` flags back to LatentSearch via the existing import endpoint.

Search on Unraid works immediately after the offload run — the vectors are
already in its local Qdrant and the DB rows are marked indexed.

## Prerequisites on the GPU machine

1. **Linux** (PyTorch's ROCm build is Linux-only; CUDA works on Linux and
   Windows, but ROCm is the relevant path for AMD GPUs).
2. **Python 3.13+** and **uv**.
3. A checkout of this repo:
   ```bash
   git clone https://github.com/coenvdgrinten/latent-search
   cd latent-search
   uv sync
   ```
4. Network reachability to both:
   - the LatentSearch HTTP port (default `8000` on the Unraid box), and
   - the Qdrant port (default `6333` on the Unraid box).

## AMD GPU setup (ROCm)

The RX 9070XT is RDNA 4 (gfx1200), which is newer than ROCm's officially
blessed list. In practice it works for inference via the standard
`HSA_OVERRIDE_GFX_VERSION` workaround. No code changes are needed —
PyTorch's ROCm build exposes the GPU through the `torch.cuda` API, which
`device.py` already auto-detects.

1. Install the PyTorch ROCm wheel (follow the current instructions at
   pytorch.org for your ROCm version).
2. Set the override before running the command:
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=12.0.0   # try 11.0.0 on older ROCm
   ```
3. Verify the GPU is visible:
   ```bash
   uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

If the GPU produces NaN vectors (the same failure mode the
`PYTORCH_DISABLE_AVX512_BF16_MATMUL` env var guards against on CPU), the
`_validate_vector` check in `vector_db.py` will reject them loudly — a
misconfigured GPU fails safe rather than silently corrupting Qdrant.

## NVIDIA GPU setup (CUDA)

Nothing special — install the PyTorch CUDA wheel and `torch.cuda.is_available()`
returns `True`. `device.py` picks it up automatically.

## Forcing CPU (for testing)

Set `LS_DEVICE=cpu` to force all ML services onto CPU, bypassing
auto-detection. Useful for validating the pipeline on a non-GPU machine or
when the GPU is busy.

## Running the offload

```bash
# Run the offload (VLM + CLIP + BGE, write to remote Qdrant + remote LatentSearch):
./manage offload_index \
  --api-url http://unraid-ip:8000 \
  --api-user admin \
  --api-password '...' \
  --qdrant-url http://unraid-ip:6333 \
  --qdrant-api-key '...' \
  --batch-size 100
```

> **Note on `--qdrant-url`**: the default derives from `--api-url` by swapping
> the port to 6333. If your LatentSearch is behind a reverse proxy (e.g.
> Cloudflare Tunnel) but Qdrant is only exposed on the LAN, you must pass
> `--qdrant-url` explicitly with the LAN IP.

### Useful flags

| Flag | Purpose |
|------|---------|
| `--limit N` | Only process N records (for testing). |
| `--dry-run` | Do everything except the Qdrant upsert and the import POST. |
| `--skip-vlm` | Skip VLM captioning; only generate embeddings from existing captions. |
| `--skip-sparse` | Skip SPLADE sparse encoding (needed when torch < 2.6 blocks `.bin` weights). |
| `--reprocess` | Process already-indexed records too (rebuild vectors). |
| `--qdrant-url` | Override Qdrant URL (defaults to `<api-url host>:6333`). |
| `--qdrant-api-key` | Qdrant API key (required if Qdrant has auth enabled). |
| `--qdrant-collection` | Override collection name (default `media_embeddings`). |

## Testing the pipeline (layered)

Validate the pipeline cheapest-first before committing to a full GPU run:

1. **Unit tests** (no GPU, no network):
   ```bash
   ./manage test latent_search.server.indexing.tests.test_offload_index
   ```
   Mocks the HTTP transport and all ML services; validates wiring.

2. **CPU dry-run against a real Unraid instance**:
   ```bash
   LS_DEVICE=cpu ./manage offload_index \
     --api-url http://unraid-ip:8000 \
     --api-user admin --api-password '...' \
     --limit 5 --dry-run
   ```
   Validates auth, export fetch, image download, model loading, and caption
   construction — without writing anything.

3. **CPU real run, tiny subset**:
   ```bash
   LS_DEVICE=cpu ./manage offload_index \
     --api-url http://unraid-ip:8000 \
     --api-user admin --api-password '...' \
     --limit 5
   ```
   Vectors actually land in Qdrant; DB rows get marked indexed. Confirm
   those 5 photos are searchable in the Unraid UI.

4. **GPU run**: drop `LS_DEVICE=cpu`, set `HSA_OVERRIDE_GFX_VERSION`, run
   `--limit 5` first, then the full batch.

## Resumability

The command is idempotent. `--pending_only=true` (the default unless
`--reprocess` is passed) means already-indexed records are skipped on
restart. The import endpoint upserts by `file_path`, so partial write-backs
merge cleanly. You can Ctrl+C and re-run freely.

## Endpoints used

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/login/` | Session login (cookie-based). |
| GET | `/api/stats` | Auth probe. |
| GET | `/api/export_library?pending_only=true` | NDJSON record list. |
| GET | `/api/media/<id>/image` | Raw image bytes by primary key. |
| POST | `/api/import_library` | NDJSON write-back (captions, vector_id, is_indexed). |

All endpoints require authentication. The offload command logs in once and
carries the session cookie.
