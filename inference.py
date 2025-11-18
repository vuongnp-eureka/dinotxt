#!/usr/bin/env python3
"""
DINOtxt Inference Script

This script demonstrates how to load and run inference with DINOtxt model using local weights.

Usage:
    python inference.py \
        --backbone_weights /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --dinotxt_weights /path/to/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth \
        --bpe_path /path/to/bpe_simple_vocab_16e6.txt.gz \
        --image_url https://dl.fbaipublicfiles.com/dinov2/images/example.jpg \
        --texts "photo of dogs" "photo of a chair" "photo of a bowl"
"""

import argparse
import urllib.request
from pathlib import Path
from typing import List, Union

import torch
from PIL import Image

from dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l, make_classification_eval_transform


def load_image_from_url(url: str) -> Image.Image:
    """Load an image from a URL."""
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


def load_image_from_path(path: str) -> Image.Image:
    """Load an image from a local file path."""
    return Image.open(path).convert("RGB")


def infer_dinotxt(
    backbone_weights: str,
    dinotxt_weights: str,
    bpe_path: str,
    image: Union[str, Image.Image],
    texts: List[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    normalize_features: bool = True,
):
    """
    Load DINOtxt model and run inference.
    
    Args:
        backbone_weights: Path to backbone weights file
        dinotxt_weights: Path to DINOtxt weights file
        bpe_path: Path to BPE vocabulary file
        image: Image URL, local path, or PIL Image
        texts: List of text queries
        device: Device to run inference on ('cuda' or 'cpu')
        normalize_features: Whether to normalize features before computing similarity
    
    Returns:
        Dictionary containing:
            - similarity: Similarity scores between texts and image
            - image_features: Image features (normalized if normalize_features=True)
            - text_features: Text features (normalized if normalize_features=True)
    """
    print(f"Loading DINOtxt model...")
    print(f"  Backbone weights: {backbone_weights}")
    print(f"  DINOtxt weights: {dinotxt_weights}")
    print(f"  BPE vocab: {bpe_path}")
    
    # Load model and tokenizer
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
        pretrained=True,
        backbone_weights=backbone_weights,
        weights=dinotxt_weights,
        bpe_path=bpe_path,
    )
    
    # Move model to device
    model = model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    
    # Load and preprocess image
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            print(f"Loading image from URL: {image}")
            img_pil = load_image_from_url(image)
        else:
            print(f"Loading image from path: {image}")
            img_pil = load_image_from_path(image)
    else:
        img_pil = image
    
    print(f"Image size: {img_pil.size}")
    
    # Preprocess image
    image_preprocess = make_classification_eval_transform()
    image_tensor = torch.stack([image_preprocess(img_pil)], dim=0).to(device)
    
    # Tokenize texts
    print(f"Tokenizing {len(texts)} texts...")
    tokenized_texts_tensor = tokenizer.tokenize(texts).to(device)
    
    # Run inference
    print("Running inference...")
    with torch.autocast(device, dtype=torch.float16 if device == "cuda" else torch.float32):
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(tokenized_texts_tensor)
    
    # Normalize features
    if normalize_features:
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity
    similarity = (text_features.cpu().float().numpy() @ image_features.cpu().float().numpy().T)
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    for i, text in enumerate(texts):
        print(f"  '{text}': {similarity[i, 0]:.4f}")
    print("="*60)
    
    return {
        "similarity": similarity,
        "image_features": image_features.cpu().numpy(),
        "text_features": text_features.cpu().numpy(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run DINOtxt inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--backbone_weights",
        type=str,
        required=True,
        help="Path to backbone weights file (dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth)"
    )
    parser.add_argument(
        "--dinotxt_weights",
        type=str,
        required=True,
        help="Path to DINOtxt weights file (dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth)"
    )
    parser.add_argument(
        "--bpe_path",
        type=str,
        required=True,
        help="Path to BPE vocabulary file (bpe_simple_vocab_16e6.txt.gz)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="https://dl.fbaipublicfiles.com/dinov2/images/example.jpg",
        help="Image URL or local path"
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        default=["photo of dogs", "photo of a chair", "photo of a bowl", "photo of a tupperware"],
        help="Text queries to compare with image"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on ('cuda' or 'cpu')"
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Don't normalize features before computing similarity"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    for path_name, path_value in [
        ("backbone_weights", args.backbone_weights),
        ("dinotxt_weights", args.dinotxt_weights),
        ("bpe_path", args.bpe_path),
    ]:
        if not Path(path_value).exists():
            # Check if it's a URL
            if not (path_value.startswith("http://") or path_value.startswith("https://")):
                raise FileNotFoundError(f"{path_name} not found: {path_value}")
    
    # Run inference
    results = infer_dinotxt(
        backbone_weights=args.backbone_weights,
        dinotxt_weights=args.dinotxt_weights,
        bpe_path=args.bpe_path,
        image=args.image,
        texts=args.texts,
        device=args.device,
        normalize_features=not args.no_normalize,
    )
    
    return results


if __name__ == "__main__":
    main()

