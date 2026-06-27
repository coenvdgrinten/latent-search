FROM python:3.12-alpine AS builder

# Install build tools needed for compiling wheels (C++ compiler, BLAS, etc.)
RUN apk add --no-cache \
    build-base \
    gcc \
    g++ \
    musl-dev \
    openblas-dev \
    jpeg-dev \
    zlib-dev \
    libpng-dev

WORKDIR /build

COPY pyproject.toml ./

RUN python -c "\
import tomllib; \
specs = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; \
print('\n'.join(specs))" > requirements.txt

RUN pip install --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir \
    -r requirements.txt

RUN apk del --purge gcc musl-dev jpeg-dev zlib-dev libpng-dev && \
    rm -rf /var/cache/apk/*

FROM python:3.12-alpine

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_DISABLE_AVX512_BF16_MATMUL=1 \
    PYTHONPATH=/app/src:/app

RUN apk add --no-cache \
    curl \
    libjpeg-turbo \
    libwebp \
    libpng

WORKDIR /app

COPY --from=builder /install /usr/local

COPY manage.py manage ./
COPY src/ ./src/

RUN mkdir -p /app/media

RUN addgroup -S app && adduser -S -G app appuser && \
    chown -R appuser:app /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/admin/login/ || exit 1

USER appuser

CMD ["sh", "-c", "python manage.py migrate --noinput && \
    python manage.py collectstatic --noinput && \
    exec gunicorn latent_search.server.config.wsgi:application \
        --chdir /app/src \
        --bind 0.0.0.0:8000 \
        --workers 1 --threads 4 \
        --timeout 300 \
        --access-logfile - --error-logfile -"]
