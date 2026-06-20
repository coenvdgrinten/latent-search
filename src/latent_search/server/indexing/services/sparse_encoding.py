"""Service for generating learned sparse embeddings using SPLADE.

Uses Sentence Transformers' SparseEncoder interface with a pretrained
SPLADE model. Produces token-level weight vectors that Qdrant can use
for lexical matching — significantly outperforming BM25 while keeping
the same sparse-vector interface.

SPLADE models expand queries/documents into weighted vocabulary indices,
catching both exact keyword matches and semantically related terms that
dense embeddings alone might undershoot.
"""

import logging

import torch
from sentence_transformers import SparseEncoder

from latent_search.server.indexing.services.device import get_device

logger = logging.getLogger(__name__)

DEFAULT_SPARSE_MODEL_ID = "naver/splade-v3-distilbert"


class SparseEncodingService:
    """
    Service for generating sparse embeddings using SPLADE.

    Uses naver/splade-v3-distilbert (67M params), trained on MS MARCO
    passage retrieval. Balances quality and CPU speed well.

    Unlike dense embeddings, SPLADE produces a sparse vector over the
    tokenizer vocabulary — non-zero entries are token indices with learned
    importance weights. This gives us keyword-like precision with neural
    query expansion.
    """

    def __init__(self, model_id: str = DEFAULT_SPARSE_MODEL_ID):
        self.model_id = model_id
        self._model: SparseEncoder | None = None
        self.device = get_device()

    @property
    def model(self) -> SparseEncoder:
        """Lazy-load the SPLADE model on first use."""
        if self._model is None:
            logger.info(f"Loading sparse encoder model '{self.model_id}'")
            self._model = SparseEncoder(self.model_id, device=self.device)
        return self._model

    def encode_document(
        self, text: str
    ) -> dict[str, list[int] | list[float]]:
        """
        Encode a single document/caption into a Qdrant-compatible sparse vector.

        Returns a dict with 'indices' (token positions) and 'values' (weights),
        which is the format Qdrant expects for sparse vectors.

        Zero-valued entries are filtered out naturally by the conversion.
        """
        embedding = self.model.encode_document(text)
        return _tensor_to_qdrant_sparse(embedding)

    def batch_encode_documents(
        self, texts: list[str]
    ) -> list[dict[str, list[int] | list[float]]]:
        """
        Batch-encode multiple documents into sparse vectors.

        Useful for indexing batches of captions efficiently.
        """
        if not texts:
            return []

        embeddings = self.model.encode_document(texts)
        return [_tensor_to_qdrant_sparse(row) for row in embeddings]


def _tensor_to_qdrant_sparse(
    tensor: torch.Tensor,
) -> dict[str, list[int] | list[float]]:
    """Convert a 1-D tensor of token scores to Qdrant sparse vector format.

    Handles both dense and sparse (CSR/COO) inputs. SPLADE models return
    COO-encoded sparse tensors; extracting indices + values directly avoids
    densification which would allocate a huge vocab-sized array.
    """
    # Handle batch output (2D) — take first row
    if tensor.ndim == 2:
        tensor = tensor.squeeze(0)

    # If already a sparse tensor, extract coords/values directly.
    if tensor.is_sparse:
        coo = tensor.coalesce().to_sparse()
        indices = coo.indices()[0].cpu().tolist()
        values = coo.values().cpu().tolist()
    else:
        nonzero_mask = tensor != 0
        indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
        values = tensor[nonzero_mask]
        indices = indices.cpu().tolist()
        values = values.cpu().tolist()

    return {
        "indices": indices,
        "values": values,
    }
