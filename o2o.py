# Copyright 2025 Nguyen Hoa Uit


import requests
import os
from dataclasses import dataclass, field
# import ..datasets import tokenize_ds

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser, is_torch_npu_available, is_torch_xpu_available

from datasets import load_dataset
from codes.o2o_config import O2OConfig
from codes.o2o_trainer import O2OTrainer
from codes.modeling_sd_base import DefaultO2OStableDiffusionPipeline
from codes.import_utils import is_npu_available, is_xpu_available

import io
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import math
from torchvision.transforms import functional as F
from codes.used_dataset import FilteredLaionAesthetic,FilteredLaionArt,VuvuzelaSet,SelectedPickaPic,ImagePickaPicDatasetHugging, ImageScoreDataset


def image_outputs_logger(image_data, global_step, accelerate_logger,caption='NA'):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    # image_data = iteration x bachsize
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]
    l=len(image_data)
    

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{caption}_{reward:.2f}_index{i}_length{l}"] = image.unsqueeze(0).float()
    try:
        accelerate_logger.log_images(
            result,
            step=global_step,
        )
    except:
        print("cannot log FileNotFoundError")


def collate_fn(batch):
    return tuple(zip(*batch))
    

@dataclass
class ScriptArguments:
# 
    pretrained_model: str = field(
        default="stabilityai/stable-diffusion-2-1", metadata={"help": "the pretrained model to use"}
    )
  
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
   
    hf_hub_model_id: str = field(
        default="Dev", metadata={"help": "HuggingFace repo to save model weights to"}
    )

    hf_hub_aesthetic_model_id: str = field(
        default="trl-lib/ddpo-aesthetic-predictor",
        metadata={"help": "HuggingFace model ID for aesthetic scorer model weights"},
    )
    
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={"help": "HuggingFace model filename for aesthetic scorer model weights"},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


# Mới adđ vào
    save_folder: str = field(
        default="./outputs/lora_weights/test_only", metadata={"help": "the folder to get checkpoin to use"}
    )
    load_folder: str = field(
        default="", metadata={"help": "the folder to get checkpoin to use"}
    )

    save_folder: str = field(
        default="./outputs/lora_weights/Fixing", metadata={"help": "the folder to get checkpoin to use"}
    )
    load_folder: str = field(
        default="", metadata={"help": "the folder to get checkpoin to use"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, O2OConfig))
    args, o2o_config = parser.parse_args_into_dataclasses(return_remaining_strings=True)[:2]
    o2o_config.project_kwargs = {
        "logging_dir": "./outputs/logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./outputs/",
    }


    transform = transforms.Compose(
        [
            transforms.Resize(o2o_config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(o2o_config.resolution),
            transforms.RandomHorizontalFlip() if True else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )

    print(f' o2o_config {o2o_config}' )
    print("-------------------------------------------------")
    print(f' args {args}' )

    dataset=None
    ix=o2o_config.dataset_index
    if ix==14:
        dataset= SelectedPickaPic(image_folder="./inputs/pick550/train", csv_file="./inputs/pick550/train.csv",transform=transform,reward=o2o_config.high_reward)      


    print("------------------------------------------------------------------------")
    print("Starting loading pipline -----------------------------------------------")
 
        
    if True:
        pipeline = DefaultO2OStableDiffusionPipeline(
            args.pretrained_model, pretrained_model_revision=args.pretrained_revision, use_lora=args.use_lora
        )

        if (args.load_folder!=''):
            pipeline.sd_pipeline.load_lora_weights(args.load_folder)

        print("------------------------------------------------------------------------")
        print("Creating trainer       -----------------------------------------------")
        trainer = O2OTrainer(
            dataset,
            o2o_config,
            pipeline,
            image_samples_hook=image_outputs_logger,
        )
        print("\n")
        print("------------------------------------------------------------------------")
        print("Starting training        -----------------------------------------------")


        epochs=o2o_config.num_epochs
        trainer.train(epochs=epochs)

        print("\n")
        print("------------------------------------------------------------------------")
        print("Starting saving model    -----------------------------------------------")

        model_note=o2o_config.huggingface_note
        num_epochs=o2o_config.global_step+o2o_config.num_epochs
        off_batch=o2o_config.offpolicy_sample_batch_size
        steps=o2o_config.sample_num_steps
        Tonline=o2o_config.online_multification_number
        name=f"dataset_index{o2o_config.dataset_index}_{model_note}_offbatch{off_batch}_T {Tonline}e{num_epochs}-timestep {steps}"
        
        print("------------------------------------------------------------------------")

        print(f"Saving local    -------{name}--------------------------------")

        trainer.save_pretrained(f"./outputs/{model_note}")

        print("------------------------------------------------------------------------")
        print("Saving hub    -----------------------------------------------")

        if (args.hf_hub_model_id==""):
            print("Not load to github")
        else:
            trainer.push_to_hub(model_note)
