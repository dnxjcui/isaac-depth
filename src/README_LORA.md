# LoRA Fine-tuning for Depth-Aware Isaac Model

This guide explains how to fine-tune the depth-aware Isaac model using LoRA (Low-Rank Adaptation).

## Overview

LoRA is a parameter-efficient fine-tuning technique that adds trainable low-rank matrices to existing model weights instead of training all parameters. This allows fine-tuning with much less memory and compute.

## Installation

Make sure you have the required dependencies:

```bash
pip install transformers peft accelerate datasets torch
```

## Data Format

Your training data should be in JSON format with the following structure:

```json
[
    {
        "id": "example_1",
        "conversations": [
            {
                "from": "human",
                "value": "What's in this image? <image>"
            },
            {
                "from": "gpt",
                "value": "This image shows..."
            }
        ],
        "image": "path/to/image.jpg"
    },
    ...
]
```

- `id`: Unique identifier for the example
- `conversations`: List of conversation turns
  - `from`: Role ("human"/"user" or "gpt"/"assistant")
  - `value`: The message text (use `<image>` or `<IMAGE>` as placeholder for images)
- `image`: Path to image file (relative to `image_folder` or absolute path)

## Usage

### Basic Training

```bash
python src/train_lora.py \
    --model_path ./isaac_model \
    --data_path ./data/train.json \
    --image_folder ./data/images \
    --output_dir ./checkpoints/isaac-lora \
    --lora_r 128 \
    --lora_alpha 256 \
    --use_depth True \
    --depth_checkpoint_path ./depth_anything_v2_vitl.pth \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --fp16 True \
    --save_steps 500 \
    --logging_steps 10
```

### Key Arguments

**Model Arguments:**
- `--model_path`: Path to pretrained Isaac model
- `--use_depth`: Enable depth-aware encoding (True/False)
- `--depth_checkpoint_path`: Path to DepthAnythingV2 checkpoint (optional)
- `--depth_alpha`: Depth scaling factor (default: 100.0)

**Data Arguments:**
- `--data_path`: Path to training data JSON file
- `--image_folder`: Path to folder containing images (if paths in JSON are relative)
- `--max_seq_length`: Maximum sequence length (default: 16384)

**LoRA Arguments:**
- `--lora_r`: LoRA rank (default: 128, higher = more parameters)
- `--lora_alpha`: LoRA alpha scaling (default: 256, typically 2x rank)
- `--lora_dropout`: LoRA dropout rate (default: 0.05)
- `--target_modules`: Comma-separated list of modules to apply LoRA (auto-detected if not specified)

**Training Arguments:**
- Standard Hugging Face TrainingArguments (see `--help` for full list)
- `--per_device_train_batch_size`: Batch size per device (use 1 for TensorStream)
- `--gradient_accumulation_steps`: Accumulate gradients over multiple steps
- `--learning_rate`: Learning rate (typically 1e-4 to 2e-4 for LoRA)
- `--fp16` or `--bf16`: Use mixed precision training

### Using a Config File

You can also use a JSON config file:

```json
{
    "model_path": "./isaac_model",
    "data_path": "./data/train.json",
    "image_folder": "./data/images",
    "output_dir": "./checkpoints/isaac-lora",
    "use_depth": true,
    "depth_checkpoint_path": "./depth_anything_v2_vitl.pth",
    "lora_r": 128,
    "lora_alpha": 256,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "fp16": true,
    "save_steps": 500,
    "logging_steps": 10
}
```

Then run:
```bash
python src/train_lora.py config.json
```

## LoRA Target Modules

By default, LoRA is applied to all linear layers except:
- Vision transformer (`vision_embedding.0`)
- Depth model (`depth_model`)
- Depth preprocessing (`depth_preproc`)

You can override this with `--target_modules` to specify exact modules, e.g.:
```bash
--target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
```

## Memory Considerations

- **Batch Size**: Use `per_device_train_batch_size=1` with TensorStream (batching is complex)
- **Gradient Accumulation**: Use `gradient_accumulation_steps` to simulate larger batches
- **Mixed Precision**: Use `--fp16` or `--bf16` to reduce memory usage
- **Gradient Checkpointing**: Use `--gradient_checkpointing` to trade compute for memory

## Loading Fine-tuned Model

After training, load the LoRA weights:

```python
from peft import PeftModel
from src.depth_isaac import IsaacForConditionalGeneration

# Load base model
base_model = IsaacForConditionalGeneration.from_pretrained("./isaac_model")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "./checkpoints/isaac-lora")

# Merge LoRA weights (optional, for inference)
model = model.merge_and_unload()
```

## Tips

1. **Start Small**: Begin with `lora_r=64` or `lora_r=128` to test
2. **Learning Rate**: LoRA typically uses higher learning rates (1e-4 to 2e-4)
3. **Depth**: Enable depth if your task benefits from spatial understanding
4. **Monitoring**: Use `--report_to wandb` for experiment tracking
5. **Checkpoints**: Use `--save_steps` to save checkpoints during training

## Troubleshooting

**Out of Memory:**
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`
- Use `fp16` or `bf16`

**Slow Training:**
- TensorStream batching is currently limited (batch_size=1)
- Consider using `gradient_accumulation_steps` to simulate larger batches
- Ensure you're using GPU acceleration

**Image Loading Errors:**
- Check that `image_folder` path is correct
- Verify image paths in JSON are relative to `image_folder` or absolute
- Ensure images are valid and can be opened with PIL

