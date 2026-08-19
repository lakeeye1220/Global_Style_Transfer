import torch
from torchvision import transforms
from typing import Union, Tuple, Optional
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

clip_id = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_id).to('cuda').eval()
clip_proc  = CLIPProcessor.from_pretrained(clip_id)

def get_text_features(text: str, norm: bool = True) -> torch.Tensor:

    text_inputs = clip_proc(text=text, return_tensors="pt", padding=True).to('cuda')
    text_features = clip_model.get_text_features(**text_inputs)
    if norm:
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

def get_image_features(img: torch.Tensor, norm: bool = True) -> torch.Tensor:
    if isinstance(img, Image.Image):
        clip_images = img
    else:
        clip_images = [
            imgs.detach().cpu().permute(1, 2, 0).numpy()
            for imgs in img
        ]

    clip_inputs = clip_proc(
        images=clip_images,
        return_tensors="pt"
    ).to("cuda")

    
    image_features = clip_model.get_image_features(**clip_inputs)
    if norm:
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)

    return image_features

def img_to_img_similarity(src_images, generated_images):
    src_img_features = get_image_features(src_images)
    gen_img_features = get_image_features(generated_images)
    return (src_img_features @ gen_img_features.T).mean()

def txt_to_img_similarity(text, generated_images):
    text_features    = get_text_features(text)
    gen_img_features = get_image_features(generated_images)
    return (text_features @ gen_img_features.T).mean()
