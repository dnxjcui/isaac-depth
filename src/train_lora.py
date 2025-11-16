#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Depth-Aware Isaac Model

This script implements parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)
for the depth-aware Isaac multimodal model.

Usage:
    python src/train_lora.py \
        --model_path ./isaac_model \
        --data_path ./data/train.json \
        --output_dir ./checkpoints/isaac-lora \
        --lora_r 128 \
        --lora_alpha 256 \
        --use_depth True
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint
from PIL import Image
from io import BytesIO
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "perceptron" / "huggingface"))
sys.path.insert(0, str(project_root / "Depth-Anything-V2"))

from src.depth_isaac import (
    IsaacConfig,
    IsaacForConditionalGeneration,
    IsaacProcessor,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def find_all_linear_names(model):
    """Find all linear layer names for LoRA targeting.
    
    Excludes vision tower and depth model to keep them frozen.
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    # Exclude these modules from LoRA (keep frozen)
    exclude_keywords = [
        'vision_embedding.0',  # Vision transformer (frozen)
        'depth_model',  # Depth model (frozen)
        'depth_preproc',  # Depth preprocessing (frozen)
    ]
    
    for name, module in model.named_modules():
        # Skip excluded modules
        if any(exclude_keyword in name for exclude_keyword in exclude_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            # Get the base module name (e.g., 'q_proj' from 'layers.0.self_attn.q_proj')
            lora_module_names.add(names[-1])
    
    # Remove lm_head if present (usually not trained with LoRA)
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_path: str = field(
        metadata={"help": "Path to pretrained Isaac model"}
    )
    depth_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DepthAnythingV2 checkpoint (optional)"}
    )
    use_depth: bool = field(
        default=False,
        metadata={"help": "Enable depth-aware encoding"}
    )
    depth_alpha: float = field(
        default=100.0,
        metadata={"help": "Depth scaling factor (alpha)"}
    )
    depth_input_size: int = field(
        default=518,
        metadata={"help": "Input size for depth model"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    data_path: str = field(
        metadata={"help": "Path to training data JSON file"}
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to folder containing images (if paths in JSON are relative)"}
    )
    max_seq_length: int = field(
        default=16384,
        metadata={"help": "Maximum sequence length"}
    )


@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration."""
    lora_r: int = field(
        default=128,
        metadata={"help": "LoRA rank (r)"}
    )
    lora_alpha: int = field(
        default=256,
        metadata={"help": "LoRA alpha scaling factor"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout rate"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias type: 'none', 'all', or 'lora_only'"}
    )
    target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of target modules (auto-detected if None)"}
    )


class IsaacDataset(Dataset):
    """Dataset for Isaac multimodal training data.
    
    Expected JSON format:
    [
        {
            "id": "example_1",
            "conversations": [
                {"from": "human", "value": "What's in this image? <image>"},
                {"from": "gpt", "value": "This image shows..."}
            ],
            "image": "path/to/image.jpg"  # or base64 encoded
        },
        ...
    ]
    """
    
    def __init__(
        self,
        data_path: str,
        processor: IsaacProcessor,
        image_folder: Optional[str] = None,
    ):
        self.processor = processor
        self.image_folder = Path(image_folder) if image_folder else None
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract conversations
        conversations = item.get("conversations", [])
        if not conversations:
            raise ValueError(f"Example {idx} has no conversations")
        
        # Build text with vision tokens (format: user message + assistant response)
        # For training, we want: "user: <text> <image> assistant: <response>"
        text_parts = []
        images = []
        
        # Process conversations to build the full prompt
        for i, conv in enumerate(conversations):
            role = conv.get("from", "").lower()
            value = conv.get("value", "")
            
            if role == "human" or role == "user":
                # Replace <image> tokens with vision token
                value = value.replace("<image>", self.processor.vision_token)
                value = value.replace("<IMAGE>", self.processor.vision_token)
                
                # Count vision tokens to know how many images we need
                vision_token_count = value.count(self.processor.vision_token)
                
                # Load images if present
                if "image" in item and vision_token_count > 0:
                    image_path = item["image"]
                    
                    # Handle base64 encoded images
                    if image_path.startswith("data:image"):
                        import base64
                        header, encoded = image_path.split(",", 1)
                        image_data = base64.b64decode(encoded)
                        image = Image.open(BytesIO(image_data)).convert("RGB")
                    else:
                        # Load from file
                        if self.image_folder and not os.path.isabs(image_path):
                            image_path = self.image_folder / image_path
                        else:
                            image_path = Path(image_path)
                        
                        if not image_path.exists():
                            raise FileNotFoundError(f"Image not found: {image_path}")
                        
                        image = Image.open(image_path).convert("RGB")
                    
                    # Add image for each vision token
                    for _ in range(vision_token_count):
                        images.append(image)
                
                # Format as user message
                text_parts.append(f"User: {value}")
            elif role == "gpt" or role == "assistant":
                # Format as assistant response
                text_parts.append(f"Assistant: {value}")
        
        # Combine all text parts with newlines
        full_text = "\n".join(text_parts) + "\n"
        
        # Process with processor
        inputs = self.processor(
            text=full_text,
            images=images if images else None,
            return_tensors="pt",
        )
        
        # Extract input_ids and tensor_stream
        input_ids = inputs["input_ids"].squeeze(0)  # Remove batch dim
        tensor_stream = inputs.get("tensor_stream")
        
        # Create labels: only compute loss on assistant responses
        # Strategy: tokenize user and assistant parts separately to find boundaries
        labels = input_ids.clone()
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        
        # Build user and assistant text separately
        user_text_parts = []
        assistant_text_parts = []
        for conv in conversations:
            role = conv.get("from", "").lower()
            value = conv.get("value", "")
            if role == "human" or role == "user":
                value = value.replace("<image>", self.processor.vision_token)
                value = value.replace("<IMAGE>", self.processor.vision_token)
                user_text_parts.append(f"User: {value}")
            elif role == "gpt" or role == "assistant":
                assistant_text_parts.append(f"Assistant: {value}")
        
        # Tokenize user part to find where it ends
        user_text = "\n".join(user_text_parts) + "\n"
        user_tokens = self.processor.tokenizer.encode(user_text, add_special_tokens=False)
        user_len = len(user_tokens)
        
        # Mask user input (everything before assistant response)
        # Account for potential special tokens added by processor
        if user_len < len(labels):
            labels[:user_len] = -100
        
        # Mask padding tokens
        labels[labels == pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "tensor_stream": tensor_stream,
            "labels": labels,
        }


def data_collator(features: List[Dict[str, Any]], processor: Optional[IsaacProcessor] = None) -> Dict[str, Any]:
    """Collate function for Isaac dataset.
    
    Handles TensorStream batching properly.
    Note: For now, we batch_size=1 per TensorStream since TensorStream
    batching is complex. For larger batches, you'd need to implement
    proper TensorStream batching logic.
    """
    batch = {}
    
    # Handle input_ids
    input_ids = [f["input_ids"] for f in features]
    max_len = max(len(ids) for ids in input_ids)
    
    # Get pad token id from processor if available
    pad_token_id = 0
    if processor and processor.tokenizer.pad_token_id is not None:
        pad_token_id = processor.tokenizer.pad_token_id
    
    # Pad input_ids
    padded_input_ids = []
    attention_mask = []
    for ids in input_ids:
        padding_len = max_len - len(ids)
        padded = torch.cat([
            ids,
            torch.full((padding_len,), pad_token_id, dtype=torch.long)
        ])
        padded_input_ids.append(padded)
        attention_mask.append(torch.cat([
            torch.ones(len(ids), dtype=torch.bool),
            torch.zeros(padding_len, dtype=torch.bool)
        ]))
    
    batch["input_ids"] = torch.stack(padded_input_ids)
    batch["attention_mask"] = torch.stack(attention_mask)
    
    # Handle labels
    labels = [f["labels"] for f in features]
    padded_labels = []
    for lbl in labels:
        padding_len = max_len - len(lbl)
        padded = torch.cat([
            lbl,
            torch.full((padding_len,), -100, dtype=lbl.dtype)
        ])
        padded_labels.append(padded)
    batch["labels"] = torch.stack(padded_labels)
    
    # Handle tensor_stream
    # Note: TensorStream batching is complex. For simplicity, we'll process
    # one at a time or implement proper batching later.
    tensor_streams = [f.get("tensor_stream") for f in features]
    if any(ts is not None for ts in tensor_streams):
        # For now, use the first non-None tensor_stream
        # In a production system, you'd want to batch TensorStreams properly
        batch["tensor_stream"] = next(ts for ts in tensor_streams if ts is not None)
    
    return batch


class IsaacTrainer(Trainer):
    """Custom trainer for Isaac model that handles TensorStream inputs."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with TensorStream support."""
        # Use tensor_stream if available, otherwise fall back to input_ids
        if "tensor_stream" in inputs and inputs["tensor_stream"] is not None:
            labels = inputs.pop("labels")
            # Remove input_ids when using tensor_stream (model will use tensor_stream)
            inputs.pop("input_ids", None)
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits  # (B, T, vocab_size)
            
            # Align labels with logits sequence length
            # Logits might have different length due to vision tokens
            seq_len = logits.size(1)
            if labels.size(1) != seq_len:
                # Pad or truncate labels to match logits
                if labels.size(1) < seq_len:
                    # Pad with -100 (ignore index)
                    padding = torch.full(
                        (labels.size(0), seq_len - labels.size(1)),
                        -100,
                        dtype=labels.dtype,
                        device=labels.device
                    )
                    labels = torch.cat([labels, padding], dim=1)
                else:
                    # Truncate labels
                    labels = labels[:, :seq_len]
            
            # Compute loss (standard causal LM: predict next token)
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            return (loss, outputs) if return_outputs else loss
        else:
            # Standard causal LM loss (using input_ids)
            return super().compute_loss(model, inputs, return_outputs=return_outputs)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass a JSON file, parse it
        model_args, data_args, lora_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Log on each process
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    
    # Load model and processor
    logger.info(f"Loading model from {model_args.model_path}")
    
    config = IsaacConfig.from_pretrained(model_args.model_path)
    
    # Set depth configuration
    if model_args.use_depth:
        config.use_depth = True
        config.depth_alpha = model_args.depth_alpha
        config.depth_input_size = model_args.depth_input_size
        if model_args.depth_checkpoint_path:
            config.depth_checkpoint_path = model_args.depth_checkpoint_path
        logger.info(f"Depth encoding enabled (alpha={model_args.depth_alpha})")
    else:
        config.use_depth = False
        logger.info("Depth encoding disabled")
    
    # Load processor
    processor = IsaacProcessor.from_pretrained(model_args.model_path)
    
    # Load model
    device_map = "auto" if training_args.device.type == "cuda" else None
    dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
    
    model = IsaacForConditionalGeneration.from_pretrained(
        model_args.model_path,
        config=config,
        torch_dtype=dtype,
        device_map=device_map,
    )
    
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Freeze base model
    model.model.requires_grad_(False)
    
    # Enable gradient checkpointing if requested
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # Setup LoRA
    from peft import LoraConfig, get_peft_model, TaskType
    
    # Find target modules
    if lora_args.target_modules:
        target_modules = [m.strip() for m in lora_args.target_modules.split(",")]
    else:
        target_modules = find_all_linear_names(model)
        logger.info(f"Auto-detected LoRA target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    logger.info("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )
    
    # Load dataset
    logger.info(f"Loading dataset from {data_args.data_path}")
    train_dataset = IsaacDataset(
        data_path=data_args.data_path,
        processor=processor,
        image_folder=data_args.image_folder,
    )
    
    # Create data collator with processor
    def collate_fn(features):
        return data_collator(features, processor=processor)
    
    # Initialize trainer
    trainer = IsaacTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )
    
    # Train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir) and training_args.do_train:
        checkpoint = get_last_checkpoint(training_args.output_dir)
    
    if training_args.do_train:
        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

