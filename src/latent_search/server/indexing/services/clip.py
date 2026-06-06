import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPService:
    """
    Service for generating embeddings using the OpenAI CLIP model.
    """

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self._model = None
        self._processor = None

    @property
    def model(self) -> CLIPModel:
        if self._model is None:
            model = CLIPModel.from_pretrained(self.model_id)
            if not isinstance(model, CLIPModel):
                raise TypeError("Downloaded model is not a CLIPModel")
            # Cast to Any to satisfy ty's incorrect type inference for .to()
            from typing import Any
            model_any: Any = model
            self._model = model_any.to(self.device)
        return self._model

    @property
    def processor(self) -> CLIPProcessor:
        if self._processor is None:
            self._processor = CLIPProcessor.from_pretrained(self.model_id)
        return self._processor

    def get_image_embedding(self, image_path: str) -> list[float]:
        """
        Generates a 512-dimensional embedding for the given image.
        """
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Normalize the embedding
        image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().flatten().tolist()

    def get_text_embedding(self, text: str) -> list[float]:
        """
        Generates a 512-dimensional embedding for the given text query.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(
            self.device
        )

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # Normalize the embedding
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().flatten().tolist()
