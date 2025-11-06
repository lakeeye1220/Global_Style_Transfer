import numpy as np
import os
import json
from tqdm import tqdm

from diffusers import StableDiffusionPipeline

"""This script creates the dataset to train concept vectors. 
This include generating images from prompts using Stable Diffusion 
and saving the images and labels in a folder."""


def update_concept_dict():
    concept_dict = ["vangogh-styled"]
    concept_dict = {c:i for i,c in enumerate(concept_dict)}
    return concept_dict


def repeat_ntimes(x, n):
    return [item for item in x for i in range(n)]


class DataCreator:
    def __init__(self, cfg):
        self.root_dir = cfg.root_dir
        self.image_prompt = repeat_ntimes(cfg.image_prompt, cfg.num_samples)
        self.input_prompt_and_target_concept = repeat_ntimes(cfg.input_prompt_and_target_concept, cfg.num_samples)
        self.validation_prompt_and_concept = cfg.validation_prompt_and_concept
        print(f"to create {len(self.image_prompt)} total number of samples in {cfg.root_dir}")

    def create_images(self, num_inference_steps=30):
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", 
            )
        pipe = pipe.to("cuda")
        pipe.safety_checker=None
        pipe.set_progress_bar_config(disable=True)

        os.makedirs(self.root_dir, exist_ok=True)
        for idx, prompt in tqdm(enumerate(self.image_prompt), total=len(self.image_prompt)):
            if isinstance(prompt, list) or isinstance(prompt, tuple):
                output = pipe(prompt[0], negative_prompt=prompt[1], num_inference_steps=num_inference_steps, return_dict=True)
                print("positive : ",prompt[0])
                print("negative prompt :",prompt[1])
            else:
                print("prompt : ",prompt)
                output = pipe(prompt, num_inference_steps=num_inference_steps, return_dict=True)
            image = output[0][0]
            image.save(self.root_dir+"/"+f"{idx}.jpg")

    def create_labels(self,):
        os.makedirs(self.root_dir, exist_ok=True)
        json.dump(self.input_prompt_and_target_concept, open(self.root_dir + "/labels.json", "w"))
        json.dump(self.validation_prompt_and_concept, open(self.root_dir + "/test.json", "w"))
        json.dump(update_concept_dict(), open(self.root_dir + "/concept_dict.json", "w"))

    def run(self):
        self.create_labels()
        self.create_images()


class Cfg:
    root_dir="datasets/vangogh_5"
    num_samples=5

    image_prompt = [
        "vangogh-styled",
    ]

    input_prompt_and_target_concept = [
        [
            ["a photo", ["vangogh-styled"]],
         ],
    ]

    validation_prompt_and_concept = ["a photo", ["vangogh-styled"]]


class CfgBatch:
    root_dir="datasets/photo"
    num_samples=10

    image_prompt = [
        "vangogh-styled",
        #"monet-styled",
    ]

    input_prompt_and_target_concept = [
        [
            ["a photo", ["vangogh-styled"]],
         ],
        [
            ["a photo", ["monet-styled"]],
         ],
    ]

    validation_prompt_and_concept = ["a photo", ["vangogh-styled", "monet-styled"]]


creator=DataCreator(Cfg)
creator.run()
