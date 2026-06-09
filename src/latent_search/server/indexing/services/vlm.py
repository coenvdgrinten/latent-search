"""
Vision-Language Model service for generating image captions.

Uses Qwen2.5-VL-3B for high-quality visual descriptions suitable for
semantic search indexing. Runs entirely on CPU with lazy loading.
"""

import logging
from pathlib import Path
from threading import Lock

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,  # ty: ignore[possibly-missing-import]
)

logger = logging.getLogger(__name__)

# Hugging Face model ID for Qwen2.5-VL-3B
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# System prompt optimized for search-caption generation
SYSTEM_PROMPT = (
    "You are a helpful assistant that describes images for a photo search engine. "
    "Provide a concise, factual description of the image contents. "
    "Focus on: people (count, actions, clothing), objects, scene type, "
    "setting/location type, lighting/atmosphere, notable colors, and any text visible. "
    "Keep descriptions under 200 characters. Be specific but brief."
)


class VLMService:
    """Lazy-loaded VLM service for image captioning."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._processor: AutoProcessor | None = None
        self._model: Qwen2VLForConditionalGeneration | None = None
        self._lock = Lock()

    @property
    def processor(self) -> AutoProcessor:
        """Lazy-load the processor on first access."""
        if self._processor is None:
            with self._lock:
                if self._processor is None:
                    logger.info(f"Loading VLM processor: {self.model_id}")
                    self._processor = AutoProcessor.from_pretrained(
                        self.model_id, trust_remote_code=True
                    )
        return self._processor

    @property
    def model(self) -> Qwen2VLForConditionalGeneration:
        """Lazy-load the model on first access."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info(
                        f"Loading VLM model: {self.model_id} "
                        f"(this may take a moment on first load)"
                    )
                    self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16,
                        device_map="cpu",
                        trust_remote_code=True,
                    )
                    self._model.eval()
        return self._model

    def describe(self, image_path: str | Path, max_tokens: int = 256) -> str:
        """
        Generate a concise caption describing the visual contents of an image.

        Args:
            image_path: Path to the image file.
            max_tokens: Maximum tokens for the generated caption.

        Returns:
            A short factual description of the image.
        """
        image = Image.open(image_path).convert("RGB")

        # Build conversation for the processor
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "Describe this image concisely."},
                ],
            },
        ]

        # Apply chat template and process
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_kwargs=True
        )
        inputs = self.processor(  # ty: ignore[call-non-callable]
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        # Generate caption
        with torch.no_grad():
            output_ids = self.model.generate(  # ty: ignore[invalid-argument-type]
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy decoding for consistency
            )

        # Decode only the newly generated tokens
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(
                inputs.input_ids, output_ids, strict=True
            )
        ]
        caption = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        return caption.strip()

    def describe_batch(
        self, image_paths: list[str | Path], max_tokens: int = 256
    ) -> list[str]:
        """
        Generate captions for multiple images sequentially.

        Args:
            image_paths: List of image file paths.
            max_tokens: Maximum tokens per caption.

        Returns:
            List of captions corresponding to input paths.
        """
        captions = []
        for i, path in enumerate(image_paths):
            logger.debug(f"Processing image {i+1}/{len(image_paths)}: {path}")
            try:
                caption = self.describe(path, max_tokens)
                captions.append(caption)
            except Exception as e:
                logger.error(f"Failed to describe {path}: {e}")
                captions.append("")  # Empty caption on failure
        return captions
