# Build stage — compile dependencies into isolated prefix
FROM python:3.13-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libjpeg-dev zlib1g-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml ./

# Extract dependency specs from pyproject.toml → requirements.txt
RUN python -c "\
import tomllib; \
specs = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; \
print('\n'.join(specs))" > requirements.txt

# Install with CPU-only PyTorch (avoids pulling 2 GB CUDA wheels)
RUN pip install --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir \
    -r requirements.txt

# --- Runtime stage ---
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_DISABLE_AVX512_BF16_MATMUL=1 \
    PYTHONPATH=/app/src:/app

# Minimal runtime system libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libjpeg62-turbo libwebpdemux2 libpng16-16 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY manage.py manage ./
COPY src/ ./src/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/admin/login/ || exit 1

CMD ["sh", "-c", "python manage.py migrate --noinput && \
    exec gunicorn config.wsgi:application \
        --bind 0.0.0.0:8000 \
        --workers 1 --threads 4 \
        --timeout 300 \
        --access-logfile - --error-logfile -"]
