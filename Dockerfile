# Build stage: compile Python wheels with C++ toolchain
FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml ./

RUN python -c "\
import tomllib; \
specs = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; \
print('\n'.join(specs))" > requirements.txt

RUN pip install --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --no-cache-dir \
    -r requirements.txt

# Remove build tools (Debian apt), not Alpine apk
RUN apt-get remove -y --purge build-essential gcc g++ libopenblas-dev \
    libjpeg-dev zlib1g-dev libpng-dev \
    && apt-get autoremove -y --purge \
    && rm -rf /var/lib/apt/lists/*

# Runtime stage: minimal Debian image with only runtime libs
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_DISABLE_AVX512_BF16_MATMUL=1 \
    PYTHONPATH=/app/src:/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gosu \
    libjpeg62-turbo \
    libwebp7 \
    libpng16-16t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local

COPY manage.py manage ./
COPY src/ ./src/
COPY entrypoint.sh /

RUN mkdir -p /app/media \
    && groupadd -r app && useradd -r -g app -d /app appuser \
    && chmod +x /entrypoint.sh

# Set home to /app so Path.home() resolves to a writable directory
ENV HOME=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/admin/login/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sh", "-c", "python manage.py migrate --noinput && \
    python manage.py collectstatic --noinput && \
    exec gunicorn latent_search.server.config.wsgi:application \
        --chdir /app/src \
        --bind 0.0.0.0:8000 \
        --workers 1 --threads 4 \
        --timeout 300 \
        --access-logfile - --error-logfile -"]
