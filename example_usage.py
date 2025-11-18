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
    BACKBONE_WEIGHTS = "/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    DINOTXT_WEIGHTS = "/path/to/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
    BPE_PATH = "/path/to/bpe_simple_vocab_16e6.txt.gz"
    
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
        backbone_weights=BACKBONE_WEIGHTS,
        weights=DINOTXT_WEIGHTS,
        bpe_path=BPE_PATH,
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    image_url = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
    with urllib.request.urlopen(image_url) as f:
        img_pil = Image.open(f).convert("RGB")
    
    image_preprocess = make_classification_eval_transform()
    image_tensor = torch.stack([image_preprocess(img_pil)], dim=0).to(device)
    
    # Prepare text queries
    texts = ["photo of dogs", "photo of a chair", "photo of a bowl", "photo of a tupperware"]
    tokenized_texts_tensor = tokenizer.tokenize(texts).to(device)
    
    # Run inference
    print("Running inference...")
    with torch.autocast(device, dtype=torch.float16 if device == "cuda" else torch.float32):
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(tokenized_texts_tensor)
    
    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity
    similarity = text_features.cpu().float().numpy() @ image_features.cpu().float().numpy().T
    
    # Print results
    print("\nSimilarity scores:")
    for text, score in zip(texts, similarity[:, 0]):
        print(f"  {text}: {score:.4f}")


if __name__ == "__main__":
    main()

