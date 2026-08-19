import logging
import os
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
from torchvision import transforms as transforms
import hf_compat
from diffusers import  DDIMInverseScheduler
import diffusers
from diffusers_modified.src.diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, PNDMScheduler, StableDiffusionPipelineGuide #,DDIMInverseScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPProcessor, CLIPModel

from model import model_types
from config import parse_args
from utils_model import save_model, load_model
from utils_data import get_dataloader, get_test_data, get_i2p_data
from clip_evaluate import txt_to_img_similarity
from clip_evaluate import img_to_img_similarity
from PIL import Image
import clip


def unfreeze_layers_unet(unet, condition):
    print("Num trainable params unet: ", sum(p.numel() for p in unet.parameters() if p.requires_grad))
    return unet


def show_images(images):
    images = [np.array(image) for image in images]
    images = np.concatenate(images, axis=1)
    return Image.fromarray(images)


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    yaml = YAML()
    yaml.dump(vars(args), open(os.path.join(args.output_dir, 'test_config.yaml'), 'w'))

    # Load models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    if args.scheduler == 'ddim':
        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, 
            beta_schedule="scaled_linear", 
            clip_sample=False, 
            set_alpha_to_one=False,
            num_train_timesteps=1000,
            steps_offset=1,
        )
    elif args.scheduler == 'pndm':
        scheduler = PNDMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="scheduler"
        )
    elif args.scheduler == 'ddpm':
        scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
        )
    else:
        raise NotImplementedError(args.scheduler)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    mlp=model_types[args.model_type](resolution=args.resolution//64)

    unet.set_controlnet(mlp)
    unet = load_model(unet, args.output_dir+'/unet.pth')
    for idx,module in enumerate(unet.controlnet.children()):
        for name, param in module.named_parameters():
            if 'weight' in name:
                param.data*=args.mlp_weight


    device=torch.device('cuda')

    model=StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    model_guide=StableDiffusionPipelineGuide(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
    model_guide=model_guide.to(device)

    directory = args.content_img

    jpg_files = [f for f in os.listdir(directory) if f.lower().endswith(".jpg")]
    if args.fp16:
        print('Using fp16')
        model.unet=model.unet.half()
        model.vae=model.vae.half()
        model.text_encoder=model.text_encoder.half()

    for img_idx, filename in enumerate(jpg_files):
        content_img = directory+filename
        dataloader=get_test_data(data_dir=args.train_data_dir, given_prompt=args.prompt, given_concept=args.concept, max_concept_length=100)
        evaluate(model=model, model_guide=model_guide, content=content_img, img_idx = img_idx, dataloader=dataloader, device=device, args=args)


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

    inverse_scheduler = diffusers.DDIMInverseScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,
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
        save_image_dir=args.output_dir+'/'+args.image_dir
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

def predict_cond(args, model,
                 content_latent, 
                 prompt, 
                 seed, 
                 condition, 
                 img_size,
                 num_inference_steps=50,
                 interpolator=None, 
                 negative_prompt=None,
                 clip_guide = False,
                 ):
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    if clip_guide:
        output = model(prompt=prompt, latents = content_latent, height=img_size, width=img_size,
                            num_inference_steps=num_inference_steps, 
                            generator=generator, 
                            controlnet_cond=condition,
                            controlnet_interpolator=interpolator,
                            negative_prompt=negative_prompt,
                            lambda_c =args.lambda_c,
                            lambda_s =args.lambda_s,
                            lambda_g = args.lambda_g,
                            clip_guide = clip_guide,
                            )
    else:
        output = model(prompt=prompt, latents = content_latent, height=img_size, width=img_size, 
                    num_inference_steps=num_inference_steps, 
                    generator=generator, 
                    controlnet_cond=condition,
                    controlnet_interpolator=interpolator,
                    negative_prompt=negative_prompt
                    )

    image = output[0][0]
    return image


def evaluate(model, model_guide, content , img_idx, dataloader, device, args):
    save_image_dir=args.output_dir+'/'+args.image_dir
    os.makedirs(save_image_dir, exist_ok=True)
    
    for j in range(args.num_test_samples):
        images=[]
        seed=j
        content_img = load_image(content,512).to(device=device)
        #print("content img shape :",content_img.shape)

        content_latent = ddim_inversion(args,content, num_steps=50, verify=False)
        
        for prompt, concept in zip(*dataloader):
            if concept is not None: #h = h + \delta*h
                result_img =predict_cond(args,model=model_guide, content_latent = content_latent, prompt=prompt, seed=seed, condition=concept, img_size=args.resolution,
                                                num_inference_steps=args.num_inference_steps,
                                                negative_prompt=args.negative_prompt, clip_guide = True
                                            )
                print("DDIM inversion O, content_guidance --> img sim: ",img_to_img_similarity(content_img, result_img))
                print("DDIM inversion O, content_guidance --> txt sim: ",txt_to_img_similarity(prompt, result_img))
        images.append(result_img)
                

    images=show_images(images)
    images.save(f"{save_image_dir}/c{args.lambda_c}_s{args.lambda_s}_g{args.lambda_g}_eval{img_idx}.jpg")



if __name__ == "__main__":
    main()
