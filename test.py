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
#from diffusers import  DDIMInverseScheduler
import diffusers
from diffusers_modified.src.diffusers import AutoencoderKL, DDPMScheduler, DDIMInverseScheduler,StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, PNDMScheduler, StableDiffusionPipelineGuide
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


def cvtImg(img):
    img = img.permute([0, 2, 3, 1])
    img = img - img.min()
    img = (img / img.max())
    return img.numpy().astype(np.float32)

def show_examples(x):
    plt.figure(figsize=(10, 10))
    imgs = cvtImg(x)
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')

def show_examples(x):
    plt.figure(figsize=(10, 5),dpi=200)
    imgs = cvtImg(x)
    for i in range(8):
        plt.subplot(1, 8, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')

def show_images(images):
    images = [np.array(image) for image in images]
    images = np.concatenate(images, axis=1)
    return Image.fromarray(images)

def prompt_with_template(profession, template):
    profession = profession.lower()
    custom_prompt = template.replace("{{placeholder}}", profession)
    return custom_prompt


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
    #if args.use_esd:
    #    load_model(unet, 'baselines/diffusers-nudity-ESDu1-UNET.pt')

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
    #mlp.fc1.weight.data = mlp.fc1.weight.data*100

    # scaling 후 weight 값 출력
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

    # jpg 파일만 필터링
    jpg_files = [f for f in os.listdir(directory) if f.lower().endswith(".jpg")]
    if args.fp16:
        print('Using fp16')
        model.unet=model.unet.half()
        model.vae=model.vae.half()
        model.text_encoder=model.text_encoder.half()

    # 하나씩 불러오기
    for img_idx, filename in enumerate(jpg_files):
        content_img = directory+filename
        print("content img : ",content_img)
        if args.evaluation_type=="inference_i2p":
            dataloader=get_i2p_data(data_dir=args.train_data_dir, given_prompt=args.prompt, given_concept=args.concept, max_concept_length=100)
        else:
            dataloader=get_test_data(data_dir=args.train_data_dir, given_prompt=args.prompt, given_concept=args.concept, max_concept_length=100)

        if args.evaluation_type=="eval":
            evaluate(model=model, model_guide=model_guide, content=content_img, img_idx = img_idx, dataloader=dataloader, device=device, args=args)
        elif args.evaluation_type=="interpolate":
            evaluate_interpolate(model=model, dataloader=dataloader, device=device, args=args)
        elif args.evaluation_type=="winobias":
            evaluate_inference_winobias(model=model, dataloader=dataloader, device=device, args=args)
        elif args.evaluation_type=="i2p":
            evaluate_inference_i2p(model=model, dataloader=dataloader, device=device, args=args)
        else:
            raise NotImplementedError(args.evaluation_type)


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

    inv_latents, _ = pipe(prompt="", negative_prompt="", guidance_scale=1., strength=0.45,
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

'''
def ddim_inversion(args, imgname, num_steps, verify=True):
    import torch, os
    import diffusers
    from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
    from torchvision import transforms
    import matplotlib.pyplot as plt

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # 1) Inverse scheduler: v_prediction으로 강제
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='scheduler'
    )
    # SD-2.x 호환을 위해 prediction_type을 v_prediction으로 고정
    inverse_scheduler.register_to_config(prediction_type="v_prediction")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=inverse_scheduler,
        safety_checker=None,
        torch_dtype=dtype,
    ).to(device)

    vae = pipe.vae

    # 4) VAE scaling factor를 반드시 config에서 사용
    def img_to_latents(img_tensor, vae):
        # img_tensor: (B,3,H,W) in [0,1]
        img = img_tensor * 2 - 1
        with torch.no_grad():
            posterior = vae.encode(img).latent_dist
            latents = posterior.sample()
        return latents * vae.config.scaling_factor

    # 모델 고유 해상도 자동 설정 (2) - model에 맞춰 width/height 결정
    native_size = pipe.unet.config.sample_size * 8  # e.g., 96*8=768 for SD-2.1, 64*8=512 for base
    target_size = getattr(args, "target_size", native_size)
    print("target size : ",target_size)

    # 이미지 로드 함수는 target_size로 리사이즈하도록 가정
    input_img = load_image(imgname, target_size).to(device=device, dtype=dtype)  # [0,1] 텐서
    latents = img_to_latents(input_img, vae)

    # 3) inversion과 verify 단계의 조건을 일치시킵니다.
    prompt = getattr(args, "prompt", "")
    negative_prompt = getattr(args, "negative_prompt", "")
    guidance = getattr(args, "guidance_scale", 1.0)  # inversion에서도 동일 값 사용 권장

    inv_latents, _ = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance,
        width=input_img.shape[-1],
        height=input_img.shape[-2],
        output_type='latent',
        return_dict=False,
        num_inference_steps=num_steps,
        latents=latents
    )

    if verify:
        save_image_dir = os.path.join(args.output_dir, args.image_dir)
        os.makedirs(save_image_dir, exist_ok=True)

        # 1) Forward DDIM도 v_prediction으로 통일
        fwd_sched = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder='scheduler'
        )
        fwd_sched.register_to_config(prediction_type="v_prediction")
        pipe.scheduler = fwd_sched

        image = pipe(
            prompt=prompt,                 # inversion 때와 동일
            negative_prompt=negative_prompt,
            guidance_scale=guidance,       # 동일
            num_inference_steps=num_steps,
            latents=inv_latents
        ).images[0]

        fig, ax = plt.subplots(1, 2, figsize=(8,4))
        ax[0].imshow(transforms.ToPILImage()(input_img[0])); ax[0].set_title("input")
        ax[1].imshow(image); ax[1].set_title("reconstructed")
        for a in ax: a.axis("off")
        plt.tight_layout()
        plt.savefig(f'{save_image_dir}/ddim_inversion.jpg')
        plt.close(fig)

    return inv_latents
'''
def predict_cond(args, model,
                 content_latent, 
                 content_img,
                 prompt, 
                 seed, 
                 condition, 
                 img_size,
                 num_inference_steps=50,
                 interpolator=None, 
                 negative_prompt=None,
                 content_logits=None,
                 clip_guide = False,
                 classifier_guide = False,
                 recon_guide = False
                 ):
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    if content_logits == None:
        output = model(prompt=prompt, latents = content_latent, height=img_size, width=img_size, 
                    num_inference_steps=num_inference_steps, 
                    generator=generator, 
                    controlnet_cond=condition,
                    controlnet_interpolator=interpolator,
                    negative_prompt=negative_prompt
                    )
    else:
        output = model(prompt=prompt, latents = content_latent,content_img = content_img, height=img_size, width=img_size,
                    num_inference_steps=num_inference_steps, 
                    generator=generator, 
                    controlnet_cond=condition,
                    controlnet_interpolator=interpolator,
                    negative_prompt=negative_prompt,
                    content_logits = content_logits,
                    lambda_c =args.lambda_c,
                    lambda_s =args.lambda_s,
                    lambda_g = args.lambda_g,
                    clip_guide = clip_guide,
                    classifier_guide = classifier_guide,
                    recon_guide = recon_guide
                    )
    image = output[0][0]
    return image


def evaluate(model, model_guide, content , img_idx, dataloader, device, args):
    save_image_dir=args.output_dir+'/'+args.image_dir
    os.makedirs(save_image_dir, exist_ok=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)

    for j in range(args.num_test_samples):
        images=[]
        seed=j
        content_img = load_image(content,512).to(device=device)
        content_latent = ddim_inversion(args,content, num_steps=50, verify=False)
        random_latent = torch.randn((1,4,64,64),device="cuda")
        clip_inputs = processor(images=content_img, return_tensors="pt").to(device)
        img_feat = clip_model.get_image_features(**clip_inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
         
        for prompt, concept in zip(*dataloader):
            prompt_list = args.prompt.split()
                    
            text_inputs = processor(text=prompt_list, return_tensors="pt", padding=True).to(device)
            text_feats = clip_model.get_text_features(**text_inputs)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            logit_scale = clip_model.logit_scale.exp()
            content_logits = (img_feat @ text_feats.T) * logit_scale  # (1, len(prompts))
            content_probs = torch.nn.functional.softmax(content_logits, dim=-1)#[0].detach().cpu().numpy()   # [len(prompt_list)]
            print("content prob shape :",content_probs.shape)
            if concept is not None:
                
                result_img =predict_cond(args,model=model_guide, content_latent = content_latent, content_img = content_img, prompt=prompt, seed=seed, condition=concept, img_size=args.resolution,
                                            num_inference_steps=args.num_inference_steps,
                                            negative_prompt=args.negative_prompt, content_logits = content_logits, clip_guide = True, classifier_guide = False, recon_guide=False
                                        )
                print("DDIM inversion O, NST guidance img sim: ",img_to_img_similarity(content_img, result_img))
                print("DDIM inversion O, NST guidance txt sim: ",txt_to_img_similarity(prompt, result_img))
                images.append(result_img)
                
            #else: # no guidance
                #result_img = predict_cond(args,model=model,content_latent = content_latent,content_img = None, prompt=prompt, seed=seed, condition=None, img_size=args.resolution,
                #                            num_inference_steps=args.num_inference_steps,
                #                            negative_prompt=args.negative_prompt, content_logits = None
                #                        )
                #images.append(result_img)
                #print("DDIM inversion O, No h-space guid img sim: ",img_to_img_similarity(content_img, result_img))
                #print("DDIM inversion O, No h-space guid txt sim: ",txt_to_img_similarity(prompt, result_img))


        images=show_images(images)
        images.save(f"{save_image_dir}/c{args.lambda_c}_s{args.lambda_s}_g{args.lambda_g}_eval{img_idx}.jpg")


def evaluate_interpolate(model, dataloader, device, args):
    save_image_dir=args.output_dir+'/'+args.image_dir
    os.makedirs(save_image_dir, exist_ok=True)
    for j in range(args.num_test_samples):
        images=[]
        seed=j
        for prompt, concept in zip(*dataloader):
            if concept is not None:
                for z in np.linspace(0,1,11):
                    images.append(predict_cond(model=model, prompt=prompt, seed=seed, condition=concept, img_size=args.resolution, 
                                                interpolator=lambda x,y: x+y*z,
                                                num_inference_steps=args.num_inference_steps,
                                                negative_prompt=args.negative_prompt,
                                               ))
            else:
                images.append(predict_cond(model=model, prompt=prompt, seed=seed, condition=None, img_size=args.resolution,
                                        num_inference_steps=args.num_inference_steps,
                                        negative_prompt=args.negative_prompt,
                                           ))

        images=show_images(images)
        images.save(f"{save_image_dir}/inter{j}.jpg")


def evaluate_inference_winobias(model, dataloader, device, args):
    from metrics.CLIP_classify import CLIP_classification_function, add_winobias_metrics
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    
    seed=None
    logging = []
    root_dir = os.path.join(args.output_dir, 'winobias') if not args.original_sd else os.path.join(args.output_dir, 'winobias_original_sd')
        
    root_dir = os.path.join(*[root_dir, args.image_dir])
    root_dir = os.path.join(root_dir, f'template{str(args.template_key)}')
    print(f'images saved to: {root_dir}')
    
    from winobias_cfg import professions, templates
    for profession in professions:  
        save_image_dir = os.path.join(root_dir, profession)
        os.makedirs(save_image_dir, exist_ok=True)
        global_id=0
        template_lst = templates[args.template_key]
        prompts = [prompt_with_template(profession, temp) for temp in template_lst]
        for prompt in prompts:
            print(f'creating images with prompt: {prompt}')
            for j in range(args.num_test_samples):
                for _, concept in zip(*dataloader):
                    if args.original_sd:
                        image=predict_cond(model=model, prompt=prompt, seed=seed, condition=None, img_size=args.resolution,
                                            num_inference_steps=args.num_inference_steps,
                                            negative_prompt=args.negative_prompt,
                                           )
                        image.save(f"{save_image_dir}/{global_id}.jpg")
                        
                    else:
                        if concept is not None:
                            image=predict_cond(model=model, prompt=prompt, seed=seed, condition=concept, img_size=args.resolution,
                                                num_inference_steps=args.num_inference_steps,
                                                negative_prompt=args.negative_prompt,
                                                )
                            image.save(f"{save_image_dir}/{global_id}.jpg")
                    global_id+=1
                
        df = CLIP_classification_function(save_image_dir, args.clip_attributes, model=clip_model, processor=processor, return_df=True)
        result = {'profession': profession}
        sums = df.sum().to_dict()
        result.update(sums)
        logging.append(result)
        print(result)

    logging = pd.DataFrame(logging)
    logging = add_winobias_metrics(logging.set_index('profession'))
    save_name = '_'.join([s.replace(' ', '_') for s in args.clip_attributes])
    save_name += '_result.csv'
    save_path = os.path.join(root_dir, save_name)
    logging.to_csv(save_path, index=True)
    print(f'CLIP classification results saved to {save_path}')


def evaluate_inference_i2p(model, dataloader, device, args):
    seed=None
    save_image_dir=args.output_dir+'/'+args.image_dir
    os.makedirs(save_image_dir, exist_ok=True)
    labels=[]
    
    model.set_progress_bar_config(disable=True)
    global_id=0
    for j in range(args.num_test_samples):
        for prompt, concept, concept_str in tqdm(dataloader, total=len(dataloader)*(args.num_test_samples-j)):
            image=predict_cond(model=model, prompt=prompt, seed=seed, condition=concept, img_size=args.resolution,
                                num_inference_steps=args.num_inference_steps,
                                negative_prompt=args.negative_prompt,
                               )
            image.save(f"{save_image_dir}/{global_id}.jpg")
            global_id+=1
            labels.append([prompt, concept_str])

    prompts, labels = list(zip(*labels))
    from metrics.nudenet_classify import detect_nude_and_q16
    predictions=detect_nude_and_q16(folder=save_image_dir)
    logging=pd.DataFrame({'prompt':prompts, 'label':labels, 'prediction':predictions})
    logging.to_csv(save_image_dir + '/i2p.csv', index=False)
    stats=pd.read_csv(save_image_dir + '/i2p.csv').groupby("label").prediction.mean()
    stats.to_csv(save_image_dir + '/i2p_stats.csv', index=True)


if __name__ == "__main__":
    main()
