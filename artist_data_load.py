import numpy as np
import os
import json
import glob
from tqdm import tqdm
from PIL import Image
from config import parse_args

args = parse_args()

"""This script creates the dataset to train concept vectors. 
This loads existing images (instead of generating them via Stable Diffusion)
and saves the images and labels in a folder."""

def count_images(folder, exts=("*.jpg", "*.jpeg", "*.png")):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return len(paths)


def update_concept_dict():
    concept_dict = ["vangogh-styled"]
    concept_dict = {c: i for i, c in enumerate(concept_dict)}
    return concept_dict


def repeat_ntimes(x, n):
    return [item for item in x for i in range(n)]


class DataCreator:
    def __init__(self, cfg):
        self.root_dir = cfg.root_dir
        self.source_image_dir = cfg.source_image_dir
        self.image_prompt = repeat_ntimes(cfg.image_prompt, cfg.num_samples)
        self.input_prompt_and_target_concept = repeat_ntimes(cfg.input_prompt_and_target_concept, cfg.num_samples)
        self.validation_prompt_and_concept = cfg.validation_prompt_and_concept
        print(f"to create {len(self.image_prompt)} total number of samples in {cfg.root_dir}")

    def create_images(self, image_size=(512, 512)):
        os.makedirs(self.root_dir, exist_ok=True)

        # collect existing image paths from the source folder
        exts = ("*.jpg", "*.jpeg", "*.png")
        source_paths = []
        for ext in exts:
            source_paths.extend(glob.glob(os.path.join(self.source_image_dir, ext)))
        source_paths = sorted(source_paths)

        assert len(source_paths) >= len(self.image_prompt), \
            f"need at least {len(self.image_prompt)} images in {self.source_image_dir}, found {len(source_paths)}"

        for idx in tqdm(range(len(self.image_prompt)), total=len(self.image_prompt)):
            image_path = source_paths[idx]
            image = Image.open(image_path).convert("RGB")
            image = image.resize(image_size)
            image.save(self.root_dir + "/" + f"{idx}.jpg")

    def create_labels(self,):
        os.makedirs(self.root_dir, exist_ok=True)
        json.dump(self.input_prompt_and_target_concept, open(self.root_dir + "/labels.json", "w"))
        json.dump(self.validation_prompt_and_concept, open(self.root_dir + "/test.json", "w"))
        json.dump(update_concept_dict(), open(self.root_dir + "/concept_dict.json", "w"))

    def run(self):
        self.create_labels()
        self.create_images()


class Cfg:
    root_dir = args.label_resize_train_data_dir
    source_image_dir = args.train_data_dir # downloaded dataset directory path
    num_samples = count_images(source_image_dir)

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
    root_dir = args.label_resize_train_data_dir
    source_image_dir = args.train_data_dir # downloaded dataset directory path
    num_samples = count_images(source_image_dir) 


    image_prompt = [
        "vangogh-styled",  # artist name's -styled"
    ]

    input_prompt_and_target_concept = [
        [
            ["a photo", ["vangogh-styled"]],
        ],
    ]

    validation_prompt_and_concept = ["a photo", ["vangogh-styled"]]


creator = DataCreator(Cfg)
creator.run()