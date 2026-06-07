
from django.conf import settings
from qdrant_client import QdrantClient

from latent_search.server.indexing.services.clip import CLIPService


class SearchService:
    def __init__(self) -> None:
        self.clip_service = CLIPService()
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self.collection_name = settings.QDRANT_COLLECTION

    def semantic_search(self, query: str, limit: int =24) -> list[dict]:
        """Converts a text string to an embedding and finds matching records."""
        # Generate embedding for the query
        query_embedding = self.clip_service.get_text_embedding(query)

        # Search in Qdrant
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
        ).points

        hits: list[dict] = []
        for hit in search_results:
            hits.append({
                "id": hit.id,
                "score": hit.score,
                "file_path": hit.payload.get("file_path", "") if hit.payload else "",
                "file_name": hit.payload.get("file_name", "") if hit.payload else "",
            })

        return hits