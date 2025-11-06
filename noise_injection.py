import os
from glob import glob
from PIL import Image
import torch
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

# -------------------------------
# 1. 파이프라인 준비
# -------------------------------
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32
).to(device)

vae = pipe.vae
scheduler = pipe.scheduler

# timesteps 설정
num_inference_steps = 50
scheduler.set_timesteps(num_inference_steps, device=device)

# 관심 timestep
target_timesteps = [10, 25, 40]

# -------------------------------
# 2. 폴더 준비
# -------------------------------
input_dir = "./domain_dataset2"
output_dirs = {t: f"./domain_dataset{t}" for t in target_timesteps}
for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

# -------------------------------
# 3. 이미지 로더
# -------------------------------
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),                # [0,1]
    T.Normalize([0.5], [0.5])    # [-1,1] 범위로
])

images = glob(os.path.join(input_dir, "*/*.jpg")) + glob(os.path.join(input_dir, "*/*.png"))

# -------------------------------
# 4. 이미지별로 timestep 노이즈 주입
# -------------------------------
with torch.no_grad():
    for img_path in tqdm(images, desc="Processing images"):
        img_name = os.path.basename(img_path)
        class_name = os.path.basename(os.path.dirname(img_path))

        # 저장할 각 클래스 폴더 생성
        for t in target_timesteps:
            os.makedirs(os.path.join(output_dirs[t], class_name), exist_ok=True)

        # 이미지 로드
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # (1,3,512,512)

        # VAE encode → latent
        latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor  # (1,4,64,64)

        # 각 target timestep마다 노이즈 추가
        noise = torch.randn_like(latents)
        for t in target_timesteps:
            timestep = scheduler.timesteps[t]  # t번째 step에 해당하는 값
            noised_latent = scheduler.add_noise(latents, noise, timestep)

            # 디코딩해서 noisy 이미지 복원
            noised_image = vae.decode(noised_latent / vae.config.scaling_factor).sample
            noised_image = (noised_image.clamp(-1, 1) + 1) / 2.0  # [0,1]
            noised_image = noised_image[0].permute(1, 2, 0).cpu().numpy() * 255
            noised_image = Image.fromarray(noised_image.astype("uint8"))

            # 저장
            save_path = os.path.join(output_dirs[t], class_name, img_name)
            noised_image.save(save_path)