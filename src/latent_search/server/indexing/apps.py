import logging
import threading
from typing import override

from django.apps import AppConfig

logger = logging.getLogger(__name__)

_model_ready_event = threading.Event()


def _warmup_text_embedding():
    """Load the text embedding model in a background thread."""
    try:
        from latent_search.server.indexing.services.text_embedding import (
            TextEmbeddingService,
        )

        logger.info("Warming up text embedding model...")
        service = TextEmbeddingService()
        # Force model load by accessing the property
        _ = service.model
        _model_ready_event.set()
        logger.info("Text embedding model is ready.")
    except Exception:
        logger.exception("Failed to warm up text embedding model.")
        # Still mark as ready so the UI isn't stuck forever
        _model_ready_event.set()


class IndexingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "latent_search.server.indexing"

    @override
    def ready(self):
        # Start model warmup in a daemon thread so it doesn't block Django startup
        thread = threading.Thread(target=_warmup_text_embedding, daemon=True)
        thread.start()
