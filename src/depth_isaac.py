from __future__ import annotations

from collections import defaultdict
from typing import Any, TypedDict

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image


from transformers import (
    AutoTokenizer,
    BatchFeature,
    Cache,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3PreTrainedModel,
)
from transformers.cache_utils import SlidingWindowCache, StaticCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Model
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import TensorType
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
import re

from transformers.models.siglip2.modeling_siglip2 import (
    Siglip2MLP,
)
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
from perceptron.tensorstream import (
    Event,
    Stream,
    TensorStream,
    TextType,
    VisionType,
    create_stream,
    group_streams,
)
from perceptron.tensorstream.ops import (
    compute_mrope_pos_tensor,
    modality_mask,
    reconstruct_tensor_stream_from_compact_dict,
    slice as ts_slice,
    tensor_stream_token_view,
)

# Import DepthAnythingV2 for depth estimation
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2

# Import base classes from modular_isaac
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'perceptron', 'huggingface'))
from modular_isaac import (
    PixelShuffleSiglip2VisionConfig,
    create_cumulative_seq_lengths,
    _max_from_cu,
    flash_attention_document_mask_forward,
    sdpa_document_mask_forward,
    Siglip2VariableSequenceEmbeddings,
    Siglip2VariableLengthAttention,
    IsaacSiglip2EncoderLayer,
    IsaacEncoder,
    create_pixel_shuffle_index_map,
    pixel_shuffle_varlen,
    MAX_PIXELS,
    VISION_MEAN,
    VISION_STD,
    VISION_SCALE,
    _make_writeable,
    extract_image_pil,
    get_image_size_for_max_num_patches,
    _MEAN_TENSOR,
    _STD_TENSOR,
    prepare_image_tensor,
    patchify_vision,
    process_vision_for_patches,
    precompute_inv_freq,
    precompute_cos_sin_3d,
    RopeScaling,
    IsaacConfig,
    create_text_event,
    compute_position_ids_input_ids,
    IsaacRotaryEmbedding,
    Siglip2SequenceVisionTransformer,
)

# Import torchvision for depth preprocessing
import torchvision.transforms as T


# ============================================================================
# Depth Preprocessing Module
# ============================================================================


class IsaacDepthPreproc(nn.Module):
    """Depth preprocessing module for DepthAnythingV2 input preparation."""

    def __init__(self, input_size: int = 518):
        """Initialize depth preprocessing.

        Args:
            input_size: Target input size for DepthAnythingV2 (default: 518)
        """
        super().__init__()
        self.resize = T.Resize(
            (input_size, input_size),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for depth model.

        Args:
            x: Input tensor of shape (B, C, H, W), values in 0-255 or 0-1 range

        Returns:
            Preprocessed tensor ready for DepthAnythingV2
        """
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        x = self.resize(x)
        x = self.normalize(x)
        return x


# ============================================================================
# Depth Positional Encoding (SD-VLM Style)
# ============================================================================


def depth_sincos_encoding(
    img_features: torch.Tensor, depth_features: list[torch.Tensor]
) -> torch.Tensor:
    """SD-VLM-style depth sinusoidal encoding.

    Implements depth_sincos_encoding from llava_arch.py, adapted for Isaac's format.
    Depth features are concatenated, reshaped, and encoded using sinusoidal functions.

    Args:
        img_features: Image features tensor of shape (B, L, dim)
        depth_features: List of depth feature tensors, one per image

    Returns:
        Image features with depth positional encoding added (additive fusion)
    """
    if len(depth_features) == 0:
        return img_features

    # Concatenate depth features (SD-VLM pattern)
    depth_features_cat = torch.cat(depth_features, dim=0)
    depth_features_flat = depth_features_cat.reshape(depth_features_cat.shape[0], -1)

    B, L, dim = img_features.shape
    assert dim % 2 == 0, f"embed_dim must be even, got {dim}"

    # Create position embedding tensor
    position_embedding = torch.zeros(
        B, L, dim, dtype=img_features.dtype, device=depth_features_flat.device
    )

    # Compute frequencies: omega = 1/(10000^(2i/dim))
    omega = torch.arange(dim // 2, dtype=img_features.dtype, device=depth_features_flat.device)
    omega /= dim / 2.0
    omega = 1.0 / (10000**omega)

    # Compute phase: sita = depth_features @ omega
    sita = depth_features_flat[:, :, None] @ omega[None, :].to(
        depth_features_flat.device
    ).to(depth_features_flat.dtype)

    # Compute sin and cos
    emb_sin = torch.sin(sita)
    emb_cos = torch.cos(sita)

    # Interleave sin and cos: [sin[0], cos[0], sin[1], cos[1], ...]
    position_embedding[:, :, 0::2] = emb_sin
    position_embedding[:, :, 1::2] = emb_cos

    # Additive fusion (SD-VLM pattern)
    return position_embedding.to(img_features.device) + img_features


class DepthPositionalEncoding(nn.Module):
    """Depth Positional Encoding module that converts depth maps to sinusoidal embeddings.

    Implements the depth encoding strategy from SD-VLM where depth values are
    encoded using sinusoidal functions similar to standard positional encodings.
    """

    def __init__(self, embed_dim: int, denom: float = 10000.0):
        """Initialize depth positional encoding.

        Args:
            embed_dim: Dimension of the output embeddings
            denom: Base for the sinusoidal frequency computation (default: 10000.0)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.denom = denom

    def forward(self, depth_map: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        """Compute depth positional encodings for a depth map.

        Args:
            depth_map: Depth map tensor of shape (B, H, W)
            patch_h: Target patch grid height
            patch_w: Target patch grid width

        Returns:
            Depth positional encodings of shape (B, patch_h*patch_w, embed_dim)
        """
        # Step 1: Adaptively average-pool depth to patch grid
        pooled = F.adaptive_avg_pool2d(
            depth_map.unsqueeze(1), (patch_h, patch_w)
        ).squeeze(1)

        # Step 2: Compute sinusoidal encoding per patch
        B, Hm, Wm = pooled.shape
        flat = pooled.view(B, -1, 1)  # (B, H′*W′, 1)
        d = self.embed_dim
        half_dim = d // 2

        # Compute frequencies: 10000^{-2t/d}
        freq = torch.exp(
            -torch.arange(0, half_dim, dtype=torch.float32, device=pooled.device)
            * (torch.log(torch.tensor(self.denom)) / half_dim)
        )

        # Compute phase for sinusoids
        args = flat * freq.unsqueeze(0).unsqueeze(0)  # broadcast
        sin = torch.sin(args)
        cos = torch.cos(args)
        pe = torch.cat([sin, cos], dim=-1)

        # Ensure embed_dim is correct (handle odd dimensions)
        if pe.size(-1) > d:
            pe = pe[..., :d]
        if pe.size(-1) < d:
            pe = F.pad(pe, (0, d - pe.size(-1)))

        return pe  # (B, H′*W′, embed_dim)


def make_depth_pe_for_isaac(
    depth_map: torch.Tensor,
    token_grids: torch.Tensor,
    dpe_module: DepthPositionalEncoding,
) -> torch.Tensor:
    """Create depth positional encodings aligned with Isaac's patch grid.

    Args:
        depth_map: Depth map tensor of shape (B, H, W) from DepthAnythingV2
        token_grids: Token grid sizes of shape (B, 2) with (height, width) per image
        dpe_module: DepthPositionalEncoding module

    Returns:
        Concatenated depth embeddings of shape (total_patches, embed_dim)
    """
    B = depth_map.shape[0]
    depth_pe_list = []

    for i in range(B):
        Hp, Wp = token_grids[i].tolist()
        pe_bhw = dpe_module(depth_map[i:i+1], Hp, Wp)  # (1, H′*W′, embed_dim)
        depth_pe_list.append(pe_bhw.squeeze(0))  # (H′*W′, embed_dim)

    # Concatenate all depth embeddings
    depth_pe = torch.cat(depth_pe_list, dim=0)  # (total_patches, embed_dim)
    return depth_pe


# ============================================================================
# Extended Config with Depth Support
# ============================================================================

# Extend IsaacConfig to add depth parameters
# We'll create a wrapper or extend it here
# For now, we'll access depth parameters via getattr with defaults


# ============================================================================
# Modified Processor with Raw Image Storage
# ============================================================================


class IsaacProcessor(ProcessorMixin):
    """Processor with raw image metadata storage for depth computation."""

    attributes = ["tokenizer"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        tokenizer: Qwen2Tokenizer,
        config: IsaacConfig | dict,
    ):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer

        if isinstance(config, dict):
            config = IsaacConfig(**config)
        self.config = config

        # Use vision token from config
        self.vision_token = config.vision_token

        # Processing parameters
        self.max_sequence_length = config.max_sequence_length

        # Vision processing parameters
        self.patch_size = config.video_patch_size
        self.max_num_patches = config.vision_max_num_patches
        self.min_num_patches = config.vision_min_num_patches
        self.pixel_shuffle_scale = config.pixel_shuffle_scale

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Any:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def build_event_stream_simple(
        self,
        text: str,
        images: list[PIL.Image.Image] | None = None,
    ) -> Stream:
        events = []
        # Process text and images
        # Find all occurrences of vision token

        pattern = re.escape(self.vision_token)
        parts = re.split(f"({pattern})", text)  # Keep the delimiter in the result

        image_idx = 0
        for current_time, part in enumerate(parts):
            if part == self.vision_token:
                # Validate that images are provided when vision token is found
                if images is None:
                    raise ValueError(
                        f"Found vision token {self.vision_token} in text but no images were provided."
                    )
                # Replace vision token with image event
                if image_idx < len(images):
                    # Create vision event from PIL image
                    image_tensor = extract_image_pil(images[image_idx])
                    if image_tensor is not None:
                        # Create a vision event with the image tensor
                        vision_event = Event(
                            data=image_tensor.unsqueeze(
                                0
                            ),  # HWC format from extract_image_pil
                            type=VisionType.image,  # I-frame
                            time=(current_time, current_time),
                        )
                        events.append(vision_event)
                    image_idx += 1
            elif part:  # Non-empty text part
                # tokens = self.text_processor.tokenize(part, add_special_tokens=False)
                text_event = create_text_event(self.tokenizer, part, time=current_time)
                events.append(text_event)

        # Process vision events if any
        if any(event.type == VisionType.image for event in events):
            # Separate text and vision events for processing
            text_events = [event for event in events if event.type == TextType.text]
            vision_events = [
                event for event in events if event.type == VisionType.image
            ]

            # Process vision events using functional approach
            processed_vision_events = []
            for vision_event in vision_events:
                # Store original image for depth computation
                raw_image = vision_event.data.clone()  # Keep a copy of raw image

                # Process the vision data
                patches, dims_virtual = process_vision_for_patches(
                    vision_event.data.squeeze(0),  # Remove the extra dimension
                    patch_size=self.patch_size,
                    max_num_patches=self.max_num_patches,
                    min_num_patches=self.min_num_patches,
                    pixel_shuffle_scale=self.pixel_shuffle_scale,
                )

                # Update event with processed data
                vision_event.data = patches.unsqueeze(1)  # Add back frame dimension
                vision_event.dims_virtual = dims_virtual
                vision_event.dims_real = (
                    dims_virtual
                    if self.pixel_shuffle_scale == 1
                    else [
                        dims_virtual[0],
                        dims_virtual[1] * self.pixel_shuffle_scale,
                        dims_virtual[2] * self.pixel_shuffle_scale,
                    ]
                )
                vision_event.idx_range = (0, math.prod(dims_virtual))

                # Flatten the patches
                vision_event.data = vision_event.data.reshape(
                    -1, vision_event.data.shape[-1]
                )

                # Store raw image as metadata for depth computation
                if not hasattr(vision_event, 'metadata'):
                    vision_event.metadata = {}
                vision_event.metadata['raw_image'] = raw_image.squeeze(0)  # (H, W, C)

                processed_vision_events.append(vision_event)

            events = text_events + processed_vision_events

        # Create stream without scheduling (events already in order)
        return create_stream(
            events, priority=[TextType.text, VisionType.image], schedule=True
        )

    def __call__(
        self,
        text: str | list[str],
        images: PIL.Image.Image | list[PIL.Image.Image] | None = None,
        return_tensors: str | TensorType | None = TensorType.PYTORCH,
        **kwargs,
    ) -> BatchFeature:
        """
        Process text and images into TensorStream format.
        Args:
            text: Input text or list of texts with vision tokens
            images: PIL image or list of images (optional)
            return_tensors: Format for output tensors

        Returns:
            BatchFeature with input_ids and tensor_stream
        """
        # Normalize inputs to lists
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        if images is not None:
            if isinstance(images, PIL.Image.Image):
                images_list = [images]
            else:
                images_list = images
        else:
            images_list = None

        if len(texts) != 1:
            raise ValueError("IsaacProcessor currently supports batch_size=1")
        if images_list is not None:
            # Count vision tokens in text to validate image count
            vision_token_count = texts[0].count(self.vision_token)
            if vision_token_count != len(images_list):
                raise ValueError(
                    f"Number of {self.vision_token} tokens in text ({vision_token_count}) "
                    f"must match number of images ({len(images_list)})"
                )

        # Build event stream
        stream = self.build_event_stream_simple(
            text=texts[0],
            images=images_list,
        )

        # Create TensorStream
        tensor_stream = TensorStream([stream])

        # Slice to max length if needed
        _, T = tensor_stream.shape
        if T > self.max_sequence_length:
            tensor_stream = ts_slice(
                tensor_stream, start=T - self.max_sequence_length, end=T
            )

        # Get token view
        tokens = tensor_stream_token_view(tensor_stream)
        if return_tensors in (TensorType.PYTORCH, "pt"):
            input_ids = torch.as_tensor(tokens, dtype=torch.long)
        else:
            input_ids = tokens

        data = {
            "input_ids": input_ids,
            "tensor_stream": tensor_stream,
        }

        return BatchFeature(data=data)


# ============================================================================
# Modified Model with Depth Support
# ============================================================================


class IsaacDepthModel(Qwen3Model):
    """Isaac model with depth-aware vision encoding."""

    def __init__(self, config: IsaacConfig):
        super().__init__(config)
        text_cfg = getattr(config, "get_text_config", lambda: config)()
        self.layers = torch.nn.ModuleList(
            [
                Qwen3DecoderLayer(text_cfg, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rotary_emb = IsaacRotaryEmbedding(config, device=self.device)

        vision_cfg = config.vision_config
        # Fix: Check for None BEFORE accessing _attn_implementation
        if vision_cfg is None:
            raise ValueError("IsaacConfig should always have vision_config")
        # Use vision_attn_implementation if specified, otherwise fall back to general attn_implementation
        vision_cfg._attn_implementation = (
            config.vision_attn_implementation
            if config.vision_attn_implementation is not None
            else config._attn_implementation
        )

        hidden_dim = vision_cfg.hidden_size * (vision_cfg.pixel_shuffle_scale_factor**2)

        # Vision embedding as Sequential to match pre-trained weight structure
        # Index 0: transformer, Index 1-3: projection layers
        self.vision_embedding = nn.Sequential(
            Siglip2SequenceVisionTransformer(vision_cfg),
            nn.Linear(
                hidden_dim,
                4 * hidden_dim,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, config.hidden_size, bias=False),
        )

        # Initialize depth model and preprocessing only if use_depth is enabled
        self.use_depth = getattr(config, 'use_depth', True)
        self.depth_alpha = getattr(config, 'depth_alpha', 100.0)
        self.depth_input_size = getattr(config, 'depth_input_size', 518)

        if self.use_depth:
            # Initialize depth model
            self.depth_model = DepthAnythingV2(encoder="vitl")

            # Load pretrained DepthAnythingV2 weights if available
            depth_checkpoint_path = getattr(config, 'depth_checkpoint_path', None)
            if depth_checkpoint_path is not None:
                import os
                if os.path.exists(depth_checkpoint_path):
                    print(f"Loading DepthAnythingV2 weights from: {depth_checkpoint_path}")
                    checkpoint = torch.load(depth_checkpoint_path, map_location='cpu')
                    # Defensive state_dict extraction
                    state_dict = checkpoint
                    if "model" in checkpoint:
                        state_dict = checkpoint["model"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    self.depth_model.load_state_dict(state_dict, strict=False)
                    print("✓ DepthAnythingV2 weights loaded successfully")
                else:
                    print(f"Warning: Depth checkpoint not found at {depth_checkpoint_path}")
                    print("Depth model will use random initialization")
            else:
                print("Warning: No depth_checkpoint_path specified in config")
                print("Depth model will use random initialization")

            self.depth_model.eval()  # Keep depth model in eval mode
            for param in self.depth_model.parameters():
                param.requires_grad = False  # Freeze depth model

            # Move depth model to device (Fix: Bug #1.1)
            self.depth_model.to(self.device)

            # Initialize depth preprocessing module
            self.depth_preproc = IsaacDepthPreproc(input_size=self.depth_input_size).to(self.device)

            # Depth positional encoding module - uses LLM hidden_size (SD-VLM approach)
            # Depth is added AFTER vision projection, in LLM hidden space
            self.dpe_module = DepthPositionalEncoding(embed_dim=config.hidden_size).to(self.device)
        else:
            self.depth_model = None
            self.depth_preproc = None
            self.dpe_module = None

        # Dispatch table for TensorStream balanced embedding (text + vision)
        # Note: embed_vision is called with depth_pe separately, not through this table
        self.embed_fns = {
            TextType: self.embed_text_tokens,
            VisionType: self.embed_vision,
        }

    def embed_text_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed text tokens, squeezing singleton dimensions."""
        # Text events are shaped as (..., 1); squeeze the singleton index dim
        h = self.embed_tokens(token_ids)
        if h.dim() >= 2 and h.size(-2) == 1:
            h = h[..., 0, :]
        return h

    def _encode_depth(
        self,
        images: list[torch.Tensor],
        target_size: tuple[int, int],
        alpha: float = 100.0,
    ) -> list[torch.Tensor]:
        """Encode depth from images following SD-VLM pattern.

        Args:
            images: List of image tensors in (C, H, W) format
            target_size: Target size (H, W) for depth maps
            alpha: Scaling factor for depth normalization (default: 100.0)

        Returns:
            List of depth tensors, one per image
        """
        depths = []
        with torch.no_grad():
            for image in images:
                if image is None:
                    depth = torch.zeros(
                        1, target_size[0], target_size[1], device=self.device
                    )
                else:
                    # Ensure image is in (C, H, W) format
                    if image.dim() == 3 and image.shape[0] != 3:
                        # Assume (H, W, C) and permute
                        image = image.permute(2, 0, 1)
                    if image.dim() == 3:
                        image = image.unsqueeze(0)  # Add batch dim

                    # Preprocess for depth model
                    image_preproc = self.depth_preproc(image)

                    # Compute depth
                    depth = self.depth_model(image_preproc)  # (1, H, W) or (1, 1, H, W)

                    # Handle 4D output
                    if depth.dim() == 4 and depth.size(1) == 1:
                        depth = depth[:, 0]  # (1, H, W)

                # Resize to target size using AdaptiveAvgPool2d (SD-VLM pattern)
                depth = F.adaptive_avg_pool2d(
                    depth.unsqueeze(1), target_size
                ).squeeze(1)

                # Normalize to 0-1 (SD-VLM pattern)
                data_min = depth.min()
                data_max = depth.max()
                depth = (depth - data_min) / (data_max - data_min + 1e-9)

                # Scale by alpha
                depths.append(depth * alpha)

        return depths

    def _compute_depth_pe(
        self,
        raw_images: list[torch.Tensor] | None,
        token_grids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute depth positional encoding from raw images.

        Args:
            raw_images: List of raw image tensors (one per vision event)
            token_grids: Tensor of shape (N_events, 2) with spatial dimensions (H, W)

        Returns:
            Depth positional encoding tensor of shape (total_patches, embed_dim) or None
        """
        # Early return if depth is disabled
        if not self.use_depth:
            return None

        if raw_images is None or len(raw_images) == 0:
            return None

        # Fix: Option A - return None if any image is None (simpler than tracking indices)
        if any(img is None for img in raw_images):
            return None

        # Assert token_grids shape
        assert token_grids.shape[1] == 2, "token_grids must be (N_events, 2) = (Hp, Wp)"

        try:
            # Prepare images for depth computation
            # Expecting images in (H, W, C) format, convert to (C, H, W)
            batch_images = []
            device = token_grids.device
            for img in raw_images:
                if img.dim() == 3:  # (H, W, C)
                    img = img.permute(2, 0, 1)  # (C, H, W)
                # Ensure on correct device
                img = img.to(device)
                batch_images.append(img)

            # Get target size from token_grids (use first image's grid size)
            # For SD-VLM compatibility, we'll use a fixed target size like (24, 24)
            # but adapt to token_grids for Isaac
            target_size = tuple(token_grids[0].tolist())  # (Hp, Wp)

            # Encode depth using SD-VLM pattern
            depth_maps_list = self._encode_depth(
                batch_images, target_size, alpha=self.depth_alpha
            )

            # Stack depth maps for batch processing
            depth_maps = torch.stack(depth_maps_list, dim=0)  # (B, H, W)

            # Generate depth positional encodings aligned with token grids
            depth_pe = make_depth_pe_for_isaac(
                depth_maps, token_grids, self.dpe_module
            )  # (total_patches, embed_dim)

            return depth_pe

        except Exception as e:
            # If depth computation fails, continue without depth
            print(f"Warning: Depth computation failed: {e}. Continuing without depth.")
            return None

    def embed_vision(
        self,
        vision_tokens: tuple[torch.Tensor, torch.Tensor],
        depth_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed vision tokens using the vision encoder with optional depth information.

        SD-VLM approach: depth is added AFTER the vision tower and projector,
        in LLM hidden space, not in vision tower space.

        Args:
            vision_tokens: Tuple of (seq_patches, token_grids)
            depth_pe: Optional precomputed depth positional encoding tensor (in LLM hidden_size)

        Returns:
            Vision embeddings enhanced with depth information if depth_pe provided
        """
        seq_patches, token_grids = vision_tokens

        # Step 1: Extract vision features from transformer (index 0)
        # This is the vision tower, equivalent to SD-VLM's vision_tower(images)
        vision_features = self.vision_embedding[0]((seq_patches, token_grids))

        # Step 2: Apply vision projection layers (indices 1-3) to map to LLM hidden space
        # This is the multimodal projector, equivalent to SD-VLM's mm_projector
        for i in range(1, len(self.vision_embedding)):
            vision_features = self.vision_embedding[i](vision_features)

        # Step 3: Add depth positional encoding in LLM hidden space (SD-VLM approach)
        # This matches SD-VLM: image_features = depth_sincos_encoding(image_features, depth_features)
        # where image_features is already in LLM hidden dimension
        if depth_pe is not None:
            # Fix: Ensure device and dtype matching (Bug #1.2)
            depth_pe = depth_pe.to(
                device=vision_features.device, dtype=vision_features.dtype
            )
            vision_features = vision_features + depth_pe

        return vision_features

    def embed_stream(self, tensor_stream: TensorStream) -> torch.Tensor:
        """
        Embed each modality stream independently, preserving the original TensorStream
        structure and computing depth embeddings for vision tokens.
        """
        flat_stream = tensor_stream.flat_stream()
        per_modality_stream = group_streams(
            flat_stream, group_fn=lambda ev: ev.type, schedule=False
        )
        per_modality_compact_stream = {
            k: v.compact() for k, v in per_modality_stream.items()
        }

        # Collect per-event grids for vision tokens (H, W like dims sans time)
        # Also collect raw images for depth computation
        token_grids = defaultdict(list)
        raw_images = defaultdict(list)

        for stream in tensor_stream.streams:
            for event in stream:
                token_grids[event.type].append(event.dims(virtual=False))
                # Extract raw image data from event metadata for depth computation
                if event.type.modality == VisionType:
                    # Try to get raw image from metadata
                    if hasattr(event, 'metadata') and 'raw_image' in event.metadata:
                        raw_images[event.type].append(event.metadata['raw_image'])
                    else:
                        # No raw image available, depth won't be computed
                        raw_images[event.type].append(None)

        embedded_compact = {}
        for stream_type, modality_payload_tensor in per_modality_compact_stream.items():
            if stream_type.modality == VisionType:
                # Build a (N_events, 2) grid tensor with spatial dims only
                grids = token_grids.get(stream_type, [])
                if len(grids) == 0:
                    input_tensor = modality_payload_tensor
                    depth_pe = None
                else:
                    token_grids_tensor = torch.tensor(
                        grids, dtype=torch.long, device=tensor_stream.device
                    )[:, 1:]
                    input_tensor = (modality_payload_tensor, token_grids_tensor)

                    # Compute depth positional encoding from raw images
                    raw_imgs = raw_images.get(stream_type, None)
                    # Only compute depth if enabled and raw images available
                    if self.use_depth and raw_imgs is not None:
                        depth_pe = self._compute_depth_pe(raw_imgs, token_grids_tensor)
                    else:
                        depth_pe = None

                # Pass precomputed depth_pe to embed_vision
                # Call embed_vision directly to pass depth_pe keyword argument
                embedded_compact[stream_type] = self.embed_vision(
                    input_tensor, depth_pe=depth_pe
                )
            else:
                embedded_compact[stream_type] = self.embed_fns[stream_type.modality](
                    modality_payload_tensor
                )

        # Reconstruct a TensorStream with embedded payloads and compact
        embedded_ts = reconstruct_tensor_stream_from_compact_dict(
            tensor_stream, embedded_compact
        )
        h = embedded_ts.compact()  # (B, T, D)
        return h

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        tensor_stream: TensorStream | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        modality_tensor: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        """
        Forward pass with MRoPE position embeddings.

        Computes position embeddings once and passes them through all layers.
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Get inputs
        if tensor_stream is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both tensor_stream and inputs_embeds")
        elif tensor_stream is not None:
            # Embed TensorStream directly
            inputs_embeds = self.embed_stream(tensor_stream)
            # Create modality tensor if not provided
            if modality_tensor is None:
                modality_tensor = modality_mask(tensor_stream)
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            # Create text modality tensor if not provided
            if modality_tensor is None:
                batch_size, seq_length = input_ids.shape
                modality_tensor = torch.full(
                    (batch_size, seq_length),
                    TextType.text.value,
                    device=input_ids.device,
                    dtype=torch.long,
                )
        elif inputs_embeds is None:
            raise ValueError(
                "You have to specify either tensor_stream, input_ids or inputs_embeds"
            )

        # Create default position_ids if not provided
        if position_ids is None:
            if tensor_stream is not None:
                position_ids = compute_mrope_pos_tensor(tensor_stream)  # (B,L,3)
            else:
                position_ids = compute_position_ids_input_ids(input_ids)

        # Compute MRoPE position embeddings if we have custom rotary_emb
        cos, sin = self.rotary_emb(position_ids, modality_tensor)
        cos = cos.to(inputs_embeds.dtype)
        sin = sin.to(inputs_embeds.dtype)

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, False
            )

        # Initialize hidden states
        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
                **kwargs,
            )

            hidden_states = (
                layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            )

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = (
                    attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                )
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen3. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen3Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen3Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            diagonal_attend_mask = torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if (
                    not isinstance(past_key_values, SlidingWindowCache)
                    or sequence_length > target_length
                ):
                    sliding_attend_mask = torch.arange(
                        target_length, device=device
                    ) <= (cache_position.reshape(-1, 1) - config.sliding_window)
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        return causal_mask


class IsaacForConditionalGeneration(Qwen3ForCausalLM, GenerationMixin):
    """Isaac multimodal model for conditional generation with depth support."""

    config_class = IsaacConfig

    def __init__(self, config: IsaacConfig):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = IsaacDepthModel(config)  # Use our custom model
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Tracks rotary position offsets computed during a full forward pass so decode steps can reuse them.
        self.rope_deltas = None

        self.config = config

    def get_rope_index(
        self,
        input_ids: torch.Tensor | None,
        tensor_stream: TensorStream | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MRoPE position ids from a TensorStream (or 1D fallback).

        Returns (position_ids, rope_deltas). position_ids is (B,L,3) for MRoPE.
        rope_deltas is (B,1) used to advance positions in decode.
        """
        # tensor_stream present: compute 3D coords
        if tensor_stream is None and input_ids is None:
            raise ValueError(
                "`tensor_stream` or `input_ids` must be provided to compute rope indices"
            )

        if tensor_stream is not None:
            pos_3d = compute_mrope_pos_tensor(tensor_stream)  # (B,L,3)
        else:
            pos_3d = compute_position_ids_input_ids(input_ids)
        B, L, _ = pos_3d.shape

        # Max position per batch across the 3 planes and sequence dimension: (B,)
        m_per_batch = pos_3d.amax(dim=(1, 2))

        # Sequence lengths per batch: (B,)
        if attention_mask is None:
            seq_lens = torch.full_like(m_per_batch, L)
        else:
            seq_lens = (
                attention_mask.eq(1)
                .sum(dim=-1)
                .to(dtype=m_per_batch.dtype, device=m_per_batch.device)
            )

        rope_deltas = (m_per_batch + 1 - seq_lens).to(dtype=pos_3d.dtype).unsqueeze(1)
        return pos_3d, rope_deltas

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        tensor_stream: TensorStream | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        """
        Forward pass for conditional generation supporting both standard inputs and TensorStream.
        Uses our embed_stream approach for multimodal inputs.
        """

        # Don't compute embeddings here - let the model handle it
        if tensor_stream is not None:
            input_ids = None
        if input_ids is None and inputs_embeds is None and tensor_stream is None:
            raise ValueError(
                "Either input_ids, inputs_embeds, or tensor_stream must be provided."
            )

        # Build position ids (MRoPE) if needed and tensor_stream is available
        # During decode we reuse `self.rope_deltas` computed on the initial forward pass; `rope_delta` captures how far
        # cached rotary phases have progressed so we can advance `position_ids` without rebuilding the TensorStream.
        if position_ids is None and tensor_stream is not None:
            position_ids, self.rope_deltas = self.get_rope_index(
                input_ids, tensor_stream, attention_mask
            )
        elif position_ids is None and input_ids is not None:
            # For text inputs build position ids and modality tensor
            position_ids = compute_position_ids_input_ids(input_ids)
            if cache_position is not None and self.rope_deltas is not None:
                # Combine the incremental decode step (`cache_position`) with cached offsets so hidden states continue
                # rotating in lockstep across generation steps.
                rope_delta = (cache_position[0] + self.rope_deltas).to(input_ids.device)
            else:
                rope_delta = 0
            if cache_position is not None and not isinstance(
                rope_delta, int
            ):  # otherwise `deltas` is an int `0`
                batch_size = input_ids.shape[0]
                rope_delta = rope_delta.repeat_interleave(
                    batch_size // rope_delta.shape[0], dim=0
                )
            position_ids = position_ids.add(rope_delta)

        if tensor_stream is not None:
            modality_tensor = modality_mask(tensor_stream)
        else:
            batch_size, seq_len = input_ids.shape
            modality_tensor = torch.empty(
                batch_size, seq_len, device=position_ids.device
            ).fill_(TextType.text.value)

        outputs = self.model(
            input_ids=input_ids,
            tensor_stream=tensor_stream,
            attention_mask=attention_mask,
            position_ids=position_ids,
            modality_tensor=modality_tensor,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: list[torch.FloatTensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        tensor_stream: TensorStream | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Prepare inputs for generation, handling TensorStream inputs properly.
        """
        # Call parent preparation
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

        # Handle TensorStream for first forward pass only
        if tensor_stream is not None and (
            cache_position is None or cache_position[0] == 0
        ):
            model_inputs["tensor_stream"] = tensor_stream
        # Let forward rebuild position_ids using cached deltas during decode
        model_inputs["position_ids"] = None
        # Drop tensor_stream after step 0
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["tensor_stream"] = None
        return model_inputs

    def can_generate(self) -> bool:
        return True


__all__ = [
    "IsaacConfig",
    "IsaacDepthModel",
    "IsaacForConditionalGeneration",
    "IsaacProcessor",
    "DepthPositionalEncoding",
    "make_depth_pe_for_isaac",
    "depth_sincos_encoding",
    "IsaacDepthPreproc",
    "DepthAnythingV2",
    "Siglip2SequenceVisionTransformer",
]
