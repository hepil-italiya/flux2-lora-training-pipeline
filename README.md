# FLUX.2 LoRA Training Pipeline

A production-ready LoRA fine-tuning pipeline for FLUX.2 image editing tasks. This repository provides reproducible training and inference code for customizing FLUX.2 models with your own image data.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [License](#license)

---

## Features

- 🚀 **Memory Efficient** - Uses 4-bit quantized models and gradient checkpointing
- 🎯 **Image-to-Image Training** - Supports multi-image conditioning
- 📊 **Progress Tracking** - Real-time loss logging and checkpointing
- 🔧 **Configurable** - Easy-to-modify hyperparameters via `config.py`

---

## Requirements

| Requirement | Minimum |
|-------------|---------|
| Python | 3.10+ |
| CUDA | 11.8+ |
| GPU VRAM | 24GB+ (recommended) |
| OS | Linux (tested on Ubuntu 22.04) |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/flux2-lora-training-pipeline.git
cd flux2-lora-training-pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install diffusers transformers accelerate

# Install training dependencies
pip install peft bitsandbytes tqdm pillow

# Install optional dependencies
pip install safetensors
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from diffusers import Flux2Pipeline; print('Diffusers OK')"
```

---

## Data Preparation

### Folder Structure

Organize your training data in the following structure:

```
your_data_folder/
├── sample_001/
│   ├── prompt.txt              # Text description (required)
│   ├── input_original.png      # Original input image
│   ├── input_material.png      # Material/reference image
│   └── output.png              # Target output image
│
├── sample_002/
│   ├── prompt.txt
│   ├── input_original.png
│   ├── input_material.png
│   └── output.png
│
└── ... (more samples)
```

### File Requirements

| File | Format | Description |
|------|--------|-------------|
| `prompt.txt` | UTF-8 text | Text prompt describing the transformation |
| `input_original.png` | PNG/JPG | Original input image |
| `input_material.png` | PNG/JPG | Reference/material image for conditioning |
| `output.png` | PNG/JPG | Target output (ground truth) |

### Image Specifications

- **Resolution**: 1024×1024 pixels (default, configurable)
- **Format**: RGB (no alpha channel)
- **Aspect Ratio**: Square (1:1) recommended

### Example `prompt.txt`

```
A professional photograph showing the fabric material applied to the clothing item, 
maintaining the original garment shape with realistic texture and lighting.
```

---

## Training

### Step 1: Update Configuration

Edit `config.py` to set your paths and hyperparameters:

```python
class Config:
    # Model settings
    repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
    device = "cuda:0"
    
    # Data path (REQUIRED: update this!)
    DATA_FOLDER = "/path/to/your/training/data"
    
    # Model saving
    SAVE_PATH = "/path/to/save/checkpoints"
    MODEL_SAVE_INTERATION = 2000
    EPOCHS = 100
    
    # Hyperparameters
    LEARNING_RATE = 5e-5
    LORA_RANK = 64
    LORA_ALPHA = 32
    BATCH_SIZE = 1
```

### Step 2: Generate Text Embeddings

Pre-compute text embeddings to save VRAM during training:

```bash
python text_embedding.py
```

This will:
- Process all subfolders in `DATA_FOLDER`
- Read `prompt.txt` from each folder
- Generate `prompt_embeds.pt` and `text_ids.pt` in each folder

**Output after execution:**
```
your_data_folder/
├── sample_001/
│   ├── prompt.txt
│   ├── prompt_embeds.pt    # ← Generated
│   ├── text_ids.pt         # ← Generated
│   └── ...
```

### Step 3: Start Training

```bash
python train.py
```

**Expected Output:**

```
============================================================
FLUX 2 IMAGE-TO-IMAGE LORA TRAINING
(Using Precomputed Multimodal Embeddings)
============================================================

[INIT] Loading FLUX 2 pipeline...
[INIT] Freezing base model weights...
[INIT] Injecting LoRA into transformer...

[LORA] Trainable parameters:
trainable params: X,XXX,XXX || all params: X,XXX,XXX,XXX || trainable%: X.XX%

[TRAIN] Starting training for 100 epochs
Epoch 1/100: 100%|██████████| 500/500 [XX:XX<00:00, X.XX it/s, loss=0.XXXX]
```

### Checkpoints

Models are saved at:
- **Every N iterations**: `{SAVE_PATH}/flux_lora_epoch_X_iter_Y/`
- **Minimum loss checkpoint**: `{SAVE_PATH}/Min_Loss_Lora/`
- **Final model**: `{SAVE_PATH}/lora_final_weights/`

---

## Inference

Load your trained LoRA weights for inference:

```python
from diffusers import Flux2Pipeline
from peft import PeftModel
import torch

# Load base pipeline
pipe = Flux2Pipeline.from_pretrained(
    "diffusers/FLUX.2-dev-bnb-4bit",
    torch_dtype=torch.bfloat16
).to("cuda")

# Load LoRA weights
pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer,
    "/path/to/lora_final_weights"
)

# Generate
image = pipe(
    prompt="your prompt here",
    height=1024,
    width=1024,
    num_inference_steps=28,
    guidance_scale=3.5
).images[0]

image.save("output.png")
```

---

## Configuration

All hyperparameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `repo_id` | `diffusers/FLUX.2-dev-bnb-4bit` | Base model repository |
| `device` | `cuda:0` | GPU device |
| `LEARNING_RATE` | `5e-5` | Learning rate |
| `LORA_RANK` | `64` | LoRA rank (higher = more capacity) |
| `LORA_ALPHA` | `32` | LoRA alpha scaling |
| `BATCH_SIZE` | `1` | Training batch size |
| `EPOCHS` | `100` | Number of training epochs |
| `IMAGE_HEIGHT` | `1024` | Image height in pixels |
| `IMAGE_WEIGHT` | `1024` | Image width in pixels |
| `MAX_SEQUENCE_LENGTH` | `512` | Max text token length |

---

## Project Structure

```
flux2-lora-training-pipeline/
├── config.py              # Configuration and hyperparameters
├── text_embedding.py      # Pre-compute text embeddings
├── dataloader.py          # PyTorch dataset class
├── train.py               # Main training script
├── save_loss.py           # Loss logging utility
├── inference.ipynb        # Inference notebook
└── README.md              # This file
```

---

## Troubleshooting

### CUDA Out of Memory

- Reduce `BATCH_SIZE` to 1
- Reduce `LORA_RANK` to 32 or 16
- Ensure text embeddings are pre-computed

### Slow Training

- Enable `torch.backends.cudnn.benchmark = True` (default)
- Use SSD storage for data folder
- Reduce `IMAGE_HEIGHT` and `IMAGE_WEIGHT` if possible

### Poor Results

- Increase `EPOCHS` (try 200-500)
- Adjust `LEARNING_RATE` (try 1e-4 or 3e-5)
- Ensure diverse and high-quality training data

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [FLUX.2 by Black Forest Labs](https://blackforestlabs.ai/)
