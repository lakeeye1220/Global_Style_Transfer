# DDIM x0 approximation per timestep (PyTorch, diffusers-compatible)
# 사용법:
#   x0 = predict_x0(model_out, x_t, t, scheduler, prediction_type)
#   x0_list = collect_x0_over_timesteps(unet, x_init, cond, scheduler, prediction_type)
import argparse
import torch
from typing import Union, Tuple, Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image
#import diffusers
#import diffusers
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from transformers import CLIPModel, CLIPProcessor
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
guidance_scale = 7.5

def latent_to_pil(x0_latent: torch.Tensor, vae) -> Image.Image:
    with torch.no_grad():
        x = 1 / 0.18215 * x0_latent
        x = vae.decode(x).sample
    x = (x.clamp(-1, 1) + 1) / 2
    x = x[0].permute(1, 2, 0).cpu().float().numpy()
    x = (x * 255).round().astype("uint8")
    return Image.fromarray(x)


def _alpha_sigma_from_t(scheduler, t, dtype, device):
    # a = sqrt(alpha_bar_t), s = sqrt(1 - alpha_bar_t)
    a_bar = scheduler.alphas_cumprod.to(device=device, dtype=dtype)[t.long()]
    a = a_bar.sqrt()
    s = (1.0 - a_bar).sqrt()
    return a, s

@torch.no_grad()
def predict_x0(model_out, x_t, t, scheduler, prediction_type="epsilon"):
    """
    model_out: UNet 출력 (prediction_type에 따라 eps/v/x0)
    x_t: 현재 latent (B,C,H,W)
    t: 현재 timestep (스칼라 int 또는 (B,) 텐서)
    scheduler: DDIMScheduler (diffusers)
    prediction_type: "epsilon" | "v_prediction" | "sample"(=x0)
    """
    dtype = x_t.dtype
    device = x_t.device
    if not torch.is_tensor(t):
        t = torch.tensor([t], device=device, dtype=torch.long)
    a, s = _alpha_sigma_from_t(scheduler, t, dtype, device)  # (B,) 또는 (1,)
    # 브로드캐스트용 차원 확장
    while a.dim() < x_t.dim():
        a = a[..., None, None, None]
        s = s[..., None, None, None]

    if prediction_type in ("epsilon", "eps", "e"):
        eps = model_out
        x0 = (x_t - s * eps) / a
    elif prediction_type in ("v_prediction", "v", "v-prediction"):
        v = model_out
        # x0 = a * x_t - s * v  (from: x_t = a*x0 + s*eps, v = a*eps - s*x0)
        x0 = a * x_t - s * v
    elif prediction_type in ("sample", "x0"):
        x0 = model_out
    else:
        raise ValueError(f"unknown prediction_type: {prediction_type}")

    return x0.clamp(-1, 1)

@torch.no_grad()
def collect_x0_over_timesteps(unet, x_t, cond, scheduler, prediction_type="epsilon"):
    """
    각 timestep마다 x0 근사값을 수집. DDIM 업데이트까지 포함.
    unet: UNet2DConditionModel
    x_t: 시작 latent (x_T)
    cond: encoder_hidden_states (텍스트 조건)
    scheduler: DDIMScheduler (scheduler.set_timesteps(num_steps) 선행 필요)
    """
    x0_list = []
    for t in scheduler.timesteps:  # 내림차순
        # UNet forward
        out = unet(x_t, t, encoder_hidden_states=cond).sample
        # x0 approximation
        x0 = predict_x0(out, x_t, t, scheduler, prediction_type)
        x0_list.append(x0)
        # DDIM one step
        step = scheduler.step(out, t, x_t)
        x_t = step.prev_sample
    return x0_list

# 예시:
# scheduler.set_timesteps(50)
# cond = text_encoder(prompt_ids)[0]
# x_T = torch.randn(1, 4, H//8, W//8, device='cuda', dtype=torch.float16)
# x0_seq = collect_x0_over_timesteps(unet, x_T, cond, scheduler, prediction_type=scheduler.config.prediction_type)
def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return transforms.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents

def ddim_inversion(args,imgname, num_steps, verify= True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    inverse_scheduler = DDIMInverseScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                   scheduler=inverse_scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=dtype)
    pipe.to(device)
    vae = pipe.vae

    input_img = load_image(imgname,512).to(device=device, dtype=dtype)
    latents = img_to_latents(input_img, vae)

    inv_latents, _ = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                          width=input_img.shape[-1], height=input_img.shape[-2],
                          output_type='latent', return_dict=False,
                          num_inference_steps=num_steps, latents=latents)

    # verify
    if verify:
        save_image_dir=args.output_dir
        os.makedirs(save_image_dir, exist_ok=True)
        pipe.scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
        image = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                     num_inference_steps=num_steps, latents=inv_latents)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(transforms.ToPILImage()(input_img[0]))
        ax[1].imshow(image.images[0])
        plt.show()
        plt.savefig(f'{save_image_dir}/ddim_inversion.jpg')
    return inv_latents

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument("--output_dir",type=str,default=None,required=False)
    parser.add_argument("--prompt",type=str,default="A photo of person")
    args = parser.parse_args()
    return args
                        
if __name__ == "__main__":
    args=parse_args()
    content_img = '../content_image/image.png'
    latents = ddim_inversion(args,content_img,50,True)
    os.makedirs(args.output_dir, exist_ok=True)

    torch_dtype =torch.float32
    # scheduler.set_timesteps(50)
    # cond = text_encoder(prompt_ids)[0]
    # x_T = torch.randn(1, 4, H//8, W//8, device='cuda', dtype=torch.float16)
    #x0_seq = collect_x0_over_timesteps(unet, x_T, cond, scheduler, prediction_type=scheduler.config.prediction_type)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=torch_dtype, safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # 텍스트 인코딩
    tok = pipe.tokenizer
    te = pipe.text_encoder
    device = pipe.device
    ids = tok(
        args.prompt,
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    # 0) CLIP 로드 (SD v1.x와 동일 계열: ViT-L/14)
    clip_id = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_id).to(device).eval()
    clip_proc  = CLIPProcessor.from_pretrained(clip_id)
    to_pil = ToPILImage()

    with torch.no_grad():
        cond = te(ids)[0]  # (B,77,768)

    x0_per_step = []          # latent-space x0 기록
    x0_img_per_step = []      # 디코딩된 이미지(옵션)
    all_logits = []

    pipe.scheduler.set_timesteps(50)
    pipe.unet.eval()
    # 텍스트 리스트 준비
    text_list = args.prompt if isinstance(args.prompt, list) else [args.prompt]
    prompt_list = args.prompt.split()
            
    text_inputs = clip_proc(text=prompt_list, return_tensors="pt", padding=True).to(device)
    text_feats = clip_model.get_text_features(**text_inputs)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    #print("text features : ",text_feats.shape)

    for t in pipe.scheduler.timesteps:
        # classifier-free guidance
        with torch.no_grad():
            latent_model_input = pipe.scheduler.scale_model_input(latents, t)
            noise_uncond = pipe.unet(latent_model_input, t, encoder_hidden_states=cond).sample
            noise_text   = pipe.unet(latent_model_input, t, encoder_hidden_states=cond).sample
            noise_pred   = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # (A) 스케줄러가 계산한 x0 사용
            out = pipe.scheduler.step(noise_pred, t, latents, return_dict=True)
            x0 = out.pred_original_sample            # == x_0 추정(latent space)
            latents = out.prev_sample

            #x0_per_step.append(x0)

            # (옵션) 이미지로 디코딩해 저장
            #sf = pipe.vae.config.scaling_factor if hasattr(pipe.vae.config, "scaling_factor") else 0.18215
            img = pipe.vae.decode(x0 / 0.1825).sample  # [-1,1]
            img = (img.clamp(-1, 1) + 1) /2    # [0,1]
            save_image(img, os.path.join(args.output_dir, f"step_{t}.png"))
           

            img_tensor = img[0].cpu()
            # 텐서를 PIL 이미지로 변환
            pil_img = to_pil(img_tensor)

            # CLIP 입력 처리
            clip_inputs = clip_proc(images=pil_img, return_tensors="pt").to(device)
            img_feat = clip_model.get_image_features(**clip_inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            #print("image features : ",img_feat.shape)

            logit_scale = clip_model.logit_scale.exp()
            logits = (img_feat @ text_feats.T) * logit_scale  # (1, len(prompts))
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].detach().cpu().numpy()   # [len(prompt_list)]

            # x축 레이블: 토큰 리스트 (예: ["A","photo","of","person"])
            tokens = prompt_list

            plt.figure(figsize=(7,4))
            plt.bar(np.arange(len(tokens)), probs)
            plt.xticks(np.arange(len(tokens)), tokens, rotation=30)
            plt.ylabel("Softmax probability")
            plt.ylim(0.0, 1.0) 
            plt.title("CLIP logits (CLIP logits → softmax)")
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(args.output_dir,f"clip_logits_{t}steps.pdf"))
        
            #all_logits = np.concatenate(all_logits, axis=0)  # shape: [num_images, num_prompts]
                #x0_img_per_step.append(img)

    #os.makedirs(args.output_dir, exist_ok=True)

    # x0_img_per_step: List[Tensor], 각 텐서는 [B,3,H,W], 값 [0,1]
    '''
    for i, img in enumerate(x0_img_per_step):
        if img.dim() == 4 and img.size(0) > 1:
            for b in range(img.size(0)):
                save_image(img[b], os.path.join(args.output_dir, f"step_{i:03d}_b{b}.png"))
        else:
            # 배치가 1장이거나 [3,H,W]인 경우
            save_image(img[0] if img.dim() == 4 else img, os.path.join(args.output_dir, f"step_{i:03d}.png"))
    '''





