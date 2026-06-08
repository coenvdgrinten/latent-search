import logging
import os

import torch
from transformers import AutoModel  # ty: ignore[possibly-missing-import]

# Prevent PyTorch from using bfloat16 matmul on CPU. Modern Intel CPUs with
# AVX-512 BF16 support will auto-promote fp32 matmuls to bf16, causing NaN
# outputs in Jina CLIP v2's encoders.
os.environ.setdefault("PYTORCH_DISABLE_AVX512_BF16_MATMUL", "1")
torch.set_autocast_cpu_enabled(False)

# Transformers 5.x renamed clip_loss → contrastive_loss, but Jina CLIP v2's
# dynamically-loaded code still imports the old name.  Monkey-patch it in
# before the model loads so this survives cache resets.
try:
    from transformers.models.clip import modeling_clip

    if not hasattr(modeling_clip, "clip_loss"):
        modeling_clip.clip_loss = modeling_clip.contrastive_loss  # type: ignore[attr-defined]
except ImportError:
    pass

logger = logging.getLogger(__name__)

VECTOR_DIM = 1024


class CLIPService:
    """
    Service for generating embeddings using the Jina CLIP v2 model.

    Uses jinaai/jina-clip-v2, which supports up to 8192 tokens for text
    and outputs 1024-dimensional vectors for both text and images.

    Note: requires trust_remote_code=True because the model uses custom
    architecture code hosted on the Hugging Face repository.
    """

    def __init__(
        self,
        model_id: str = "jinaai/jina-clip-v2",
        truncate_dim: int = VECTOR_DIM,
    ):
        self.model_id = model_id
        self.truncate_dim = truncate_dim
        self._model: AutoModel | None = None

    @property
    def model(self) -> AutoModel:
        if self._model is None:
            logger.info(f"Loading model '{self.model_id}'")
            # Load in default precision, then cast everything to float32.
            # Loading with torch_dtype=float32 breaks Jina's custom loader.
            self._model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._model.to(torch.float32)
            try:
                self._fix_rope_buffers(self._model)
                self._fix_lora_dropout_masks(self._model)
            except Exception as e:
                logger.warning(f"Buffer fix skipped: {e}")
        return self._model

    @staticmethod
    def _fix_rope_buffers(model: AutoModel) -> None:
        """
        Jina CLIP v2's RoPE buffers (freqs_cos / freqs_sin) are serialised in
        bfloat16 and overflow to ~3e35 when cast to float32, producing all-NaN
        image embeddings on CPU.  Recompute them from the model config.

        Parameters mirror VisionRotaryEmbeddingFast.__init__:
          dim          = embed_dim / num_heads / 2  (half head dim)
          pt_seq_len   = pt_hw_seq_len from vision config (16 for EVA02-L14)
          ft_seq_len   = img_size // patch_size when intp_freq=True
        """
        from einops import repeat

        vm = model.vision_model
        vision_cfg = model.config.vision_config
        rope = vm.rope
        if rope is None or not (
            hasattr(rope, "freqs_cos") and hasattr(rope, "freqs_sin")
        ):
            return

        head_width: int = getattr(vision_cfg, "head_width", 64)
        half_head_dim: int = head_width // 2  # = dim parameter

        patch_size: int = vm.patch_embed.patch_size[0]
        img_size: int = vm.patch_embed.img_size[0]
        hw_seq_len: int = img_size // patch_size  # ft_seq_len (36 for 512px)
        pt_seq_len: int = getattr(vision_cfg, "pt_hw_seq_len", hw_seq_len)

        dim = half_head_dim  # 32
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(hw_seq_len).float() / hw_seq_len * pt_seq_len
        freqs = torch.einsum("..., f -> ... f", t, freqs)  # [hw, dim//2]
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)  # [hw, dim]

        # 2-D broadcast: concat h-freqs and w-freqs → [hw*hw, 2*dim]
        freqs_2d = torch.cat(
            [
                freqs[:, None, :].expand(hw_seq_len, hw_seq_len, dim),
                freqs[None, :, :].expand(hw_seq_len, hw_seq_len, dim),
            ],
            dim=-1,
        ).reshape(hw_seq_len * hw_seq_len, 2 * dim)

        device = rope.freqs_cos.device
        rope.freqs_cos = freqs_2d.cos().to(device)
        rope.freqs_sin = freqs_2d.sin().to(device)
        logger.info(
            f"Recomputed RoPE buffers: shape={rope.freqs_cos.shape}, "
            f"max_val={rope.freqs_cos.abs().max():.4f}"
        )

    @staticmethod
    def _fix_lora_dropout_masks(model: AutoModel) -> None:
        """
        The LoRA dropout masks in the text encoder's XLM-RoBERTa backbone are
        non-persistent buffers initialized as torch.ones(...) in __init__, but
        get corrupted to garbage values (near-zero, NaN) during model loading.
        In eval mode, nn.Dropout is an identity so the mask should be all-ones;
        reset every lora_dropout_mask to ones to restore correct inference.
        """
        count = 0
        for module in model.modules():
            if "lora_dropout_mask" in module._buffers:
                buf = module._buffers["lora_dropout_mask"]
                module._buffers["lora_dropout_mask"] = torch.ones_like(buf)
                count += 1
        if count:
            logger.info(f"Reset {count} lora_dropout_mask buffers to ones")

    def get_image_embedding(self, image_path: str) -> list[float]:
        """
        Generates a 1024-dimensional embedding for the given image file.
        Accepts local file paths. Raises FileNotFoundError / OSError on
        missing or unreadable files.
        """
        try:
            with torch.no_grad(), torch.amp.autocast("cpu", enabled=False):
                embeddings = self.model.encode_image(
                    [image_path], truncate_dim=self.truncate_dim
                )
        except (FileNotFoundError, OSError) as e:
            logger.error(f"Failed to open image at '{image_path}': {e}")
            raise
        return embeddings[0].tolist()

    def get_text_embedding(
        self, text: str, task: str | None = "retrieval.query"
    ) -> list[float]:
        """
        Generates a 1024-dimensional embedding for the given text.

        Use task='retrieval.query' for search queries (default).
        Use task=None when embedding document/caption text during indexing
        so query and passage vectors share the same space.
        """
        with torch.no_grad(), torch.amp.autocast("cpu", enabled=False):
            embeddings = self.model.encode_text(
                [text], truncate_dim=self.truncate_dim, task=task
            )
        return embeddings[0].tolist()
