# DINOtxt

Minimal DINOtxt library for loading and inferring DINOtxt vision-language models. This is a lightweight extraction from the full DINOv3 library, containing only the code necessary for DINOtxt inference.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE.md)

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/vuongnp-eureka/dinotxt.git
```

Or clone and install locally:

```bash
git clone https://github.com/vuongnp-eureka/dinotxt.git
cd dinotxt
pip install -e .
```

### Install from PyPI (when published)

```bash
pip install dinotxt
```

## Usage with Local Weights

```python
from dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l, make_classification_eval_transform

# Provide paths to your local weight files
model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
    backbone_weights="/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    weights="/path/to/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
    bpe_path="/path/to/bpe_simple_vocab_16e6.txt.gz"
)

import urllib
from PIL import Image

def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")

EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
img_pil = load_image_from_url(EXAMPLE_IMAGE_URL)

import torch

image_preprocess = make_classification_eval_transform()
image_tensor = torch.stack([image_preprocess(img_pil)], dim=0).cuda()
texts = ["photo of dogs", "photo of a chair", "photo of a bowl", "photo of a tupperware"]
tokenized_texts_tensor = tokenizer.tokenize(texts).cuda()
model = model.cuda()

with torch.autocast('cuda', dtype=torch.float):
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(tokenized_texts_tensor)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (
    text_features.cpu().float().numpy() @ image_features.cpu().float().numpy().T
)
print(similarity)
```

## Required Local Files

You need to download these files locally:

1. **Backbone weights**: `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
   - Download from: `https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`

2. **DINOTxt weights**: `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`
   - Download from: `https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`

3. **BPE vocab**: `bpe_simple_vocab_16e6.txt.gz`
   - Download from: `https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz`

## Quick Start

### Using the inference script:

```bash
python -m dinotxt.inference \
    --backbone_weights /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --dinotxt_weights /path/to/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth \
    --bpe_path /path/to/bpe_simple_vocab_16e6.txt.gz \
    --image /path/to/image.jpg \
    --texts "photo of dogs" "photo of a chair"
```

### Verify Installation

```bash
python -m dinotxt.test_installation
```

Or in Python:
```python
import dinotxt
print(dinotxt.__version__)
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.13.0
- torchvision >= 0.14.0
- numpy >= 1.21.0
- Pillow >= 8.0.0
- ftfy >= 6.0.0
- regex >= 2022.0.0

See `requirements.txt` for the complete list.

## Package Structure

- `layers/` - Core transformer layers (attention, blocks, FFN, etc.)
- `models/` - Vision transformer model
- `eval/text/` - Text transformer, towers, and DINOtxt model
- `hub/` - Model loading functions (now uses local paths only)
- `data/` - Image transforms
- `utils/` - Utility functions
- `thirdparty/CLIP/` - CLIP tokenizer

## Features

- 🚀 **Lightweight**: Only essential code for DINOtxt inference
- 📦 **Easy Installation**: Install directly from GitHub
- 🔧 **Local Weights**: Load models from local weight files
- 🎯 **Simple API**: Clean and intuitive interface
- ⚡ **Fast Inference**: Optimized for production use

## Changes from Original DINOv3

- ✅ Removed all URL downloading code
- ✅ Removed RMSNorm (unused)
- ✅ Removed unused SwiGLUFFN variants
- ✅ All weights must be provided as local file paths
- ✅ Simplified tokenizer to only load from local files

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Citation

If you use DINOtxt in your research, please cite the original DINOv3 paper:

```bibtex
@article{dinov3,
  title={DINOv3: Learning Robust Visual Features without Supervision},
  author={...},
  journal={...},
  year={2024}
}
```

## Troubleshooting

### Import Errors
Make sure you're using the correct Python environment:
```bash
which python
pip list | grep dinotxt
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### CUDA Issues
Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

This library is extracted from the [DINOv3](https://github.com/facebookresearch/dinov3) project by Meta AI.
