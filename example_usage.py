"""
Simple example of using DINOtxt for inference.

This is a minimal example showing how to use DINOtxt with local weights.
"""

from dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l, make_classification_eval_transform
import torch
from PIL import Image
import urllib.request


def main():
    # Paths to your local weight files
    BACKBONE_WEIGHTS = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    DINOTXT_WEIGHTS = "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
    BPE_PATH = "bpe_simple_vocab_16e6.txt.gz"
    
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
        backbone_weights=BACKBONE_WEIGHTS,
        weights=DINOTXT_WEIGHTS,
        bpe_path=BPE_PATH,
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Detect Jetson and ensure FP32 precision
    is_jetson = False
    if device == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).lower()
        is_jetson = "jetson" in device_name or "tegra" in device_name
        if is_jetson:
            print(f"Detected Jetson device: {torch.cuda.get_device_name(0)}")
            print("Converting model to FP32 for numerical stability")
            # Ensure model is in FP32 on Jetson
            model = model.float()
    
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    image_url = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
    with urllib.request.urlopen(image_url) as f:
        img_pil = Image.open(f).convert("RGB")
    
    image_preprocess = make_classification_eval_transform()
    image_tensor = torch.stack([image_preprocess(img_pil)], dim=0).to(device)
    
    # Prepare text queries
    # texts = ["photo of dogs", "photo of a chair", "photo of a bowl", "photo of a tupperware"]
    texts = ["dogs", "chair", "bowl", "tupperware"]
    tokenized_texts_tensor = tokenizer.tokenize(texts).to(device)
    
    # Run inference
    print("Running inference...")
    
    # Use FP32 on Jetson, FP16 on other CUDA devices
    if device == "cuda" and not is_jetson:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(tokenized_texts_tensor)
    else:
        # FP32 for CPU or Jetson
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(tokenized_texts_tensor)
    
    # Normalize features with numerical stability (add small epsilon to avoid division by zero)
    image_norm = image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features.norm(dim=-1, keepdim=True)
    image_features = image_features / (image_norm + 1e-8)
    text_features = text_features / (text_norm + 1e-8)
    
    # Check for NaN/Inf and handle (important for FP16 on Jetson)
    if torch.isnan(image_features).any() or torch.isinf(image_features).any():
        print("Warning: NaN/Inf detected in image_features, replacing with zeros")
        image_features = torch.where(torch.isnan(image_features) | torch.isinf(image_features), 
                                     torch.zeros_like(image_features), image_features)
    if torch.isnan(text_features).any() or torch.isinf(text_features).any():
        print("Warning: NaN/Inf detected in text_features, replacing with zeros")
        text_features = torch.where(torch.isnan(text_features) | torch.isinf(text_features), 
                                    torch.zeros_like(text_features), text_features)
    
    # Compute similarity
    similarity = text_features.cpu().float().numpy() @ image_features.cpu().float().numpy().T
    
    # Print results
    print("\nSimilarity scores:")
    for text, score in zip(texts, similarity[:, 0]):
        print(f"  {text}: {score:.4f}")


if __name__ == "__main__":
    main()

