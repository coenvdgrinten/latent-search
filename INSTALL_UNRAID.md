# Installing LatentSearch on Unraid

This guide covers running LatentSearch on Unraid via Docker Compose.

## Prerequisites

- **Unraid** with Community Apps installed.
- **Docker Compose v2** — included with recent Unraid releases. Verify with:
  ```bash
  docker compose version
  ```
- A **Qdrant** instance (bundled or existing — see below).

---

## Option A: Using the Bundled Qdrant

The `docker-compose.yml` ships with a Qdrant service out of the box. This is the simplest path for a fresh install.

```bash
cd /path/to/latent-search
cp .env.example .env
nano .env   # Set SECRET_KEY and review defaults
docker compose up --build -d
```

LatentSearch will be available at `http://<unraid-ip>:8000`.

---

## Option B: Connecting to an Existing Qdrant Container

Many Unraid users already run Qdrant as a separate community app. To point LatentSearch at it:

1. **Remove or comment out** the `qdrant:` service block in `docker-compose.yml`.
2. **Remove** the `depends_on: [qdrant]` line from the `latentsearch` service.
3. **Set** `QDRANT_URL` in `.env` to your Qdrant address:

   ```ini
   # If Qdrant runs on another container accessible via host network:
   QDRANT_URL=http://host.docker.internal:6333

   # Or use a LAN IP:
   # QDRANT_URL=http://192.168.1.100:6333
   ```

4. Make sure `extra_hosts` is present in the `latentsearch` service (it is by default):

   ```yaml
   extra_hosts:
     - "host.docker.internal:host-gateway"
   ```

   This maps `host.docker.internal` to the Docker host gateway so containers can reach services bound to the Unraid host. Without this entry, `host.docker.internal` won't resolve on Linux Docker.

5. Start:

   ```bash
   docker compose up --build -d
   ```

---

## Persisting Across Reboots

### Important: Docker Compose v2 Plugin Limitation

Installing Docker Compose via the Unraid **v2 plugin** puts binaries under `/usr/local`, which is **reset on every reboot**. You will need to reinstall the plugin after each boot.

### Recommended Approach: User Scripts Plugin

To automate recovery after reboots, use the **User Scripts** community plugin:

1. Install the **User Scripts** plugin via Community Applications.
2. Create a new script (e.g., `start-latentsearch.sh`):

   ```bash
   #!/bin/bash
   cd /path/to/latent-search
   docker compose up -d
   ```

3. Schedule it to run at startup (or manually after a reboot).

Alternatively, you can pin the Docker Compose binary outside `/usr/local` (e.g., copy it to `/boot/config/plugins/`) and reference that path directly.

---

## Indexing Media

Once the server is running, discover and index photos:

```bash
docker exec latentsearch python manage.py index_media /nc_data/<username>/files/Photos
```

### Shell Quoting Warning

Nextcloud photo paths often contain special characters — especially apostrophes in Dutch folder names like `Photo's`. Always quote paths carefully:

```bash
# CORRECT — double quotes protect the apostrophe
docker exec latentsearch python manage.py index_media "/nc_data/user/files/Photo's"

# WRONG — unquoted apostrophe breaks the shell
docker exec latentsearch python manage.py index_media /nc_data/user/files/Photo's
```

When unsure, escape with single backslash: `'Photo\'s'` or wrap the entire argument in double quotes.

---

## VLM Caption Enrichment (Optional)

After indexing, generate AI captions for richer search:

```bash
docker exec latentsearch python manage.py enrich_captions
```

Caption generation uses a large vision-language model on CPU — expect several minutes per image. Progress is checkpointed; interrupted runs resume automatically.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `QDRANT_URL: unbound variable` | Set `QDRANT_URL` in `.env` before starting. |
| Model download stalls on first run | The CLIP/VLM models (~several GB) download on first indexing. Wait patiently; they cache afterward. |
| `Permission denied` on media files | Mount volumes read-only (`:r`) and ensure the container user can traverse parent dirs. |
| Container crashes on import | Increase tmpfs size in `docker-compose.yml` if loading large models OOM-kills the process. |

For more help, open an issue on [GitHub](https://github.com/coenvdgrinten/latent-search/issues).
