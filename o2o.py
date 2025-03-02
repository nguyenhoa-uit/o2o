# Copyright 2023 metric-space, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



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
from codes.used_dataset import SelectedPickaPic,FilteredLaionArt,ImageLionArtDatasetHugging, ImageArtPaintingDataset,ImagePickaPicDatasetHugging, ImageScoreDataset, ImageScoreDatasetCSV

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        # print(f'jpeg_incompressibility images:{images[0]} \n')
        # print(f'jpeg_incompressibility prompts:{prompts} \n')
        # print(f'jpeg_incompressibility metadata:{metadata} \n')

        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        print(f'jpeg_incompressibilityc: {sizes}.. \n')

        return np.array(sizes), {}

    return _fn

# list of example prompts to feed stable diffusion
animals = [
    "cat",
    "dog",
    "horse",
    "monkey",
    "rabbit",
    "zebra"
]


class AestheticScorer(torch.nn.Module):
    """
    This model attempts to predict the aesthetic score of an image. The aesthetic score
    is a numerical approximation of how much a specific image is liked by humans on average.
    This is from https://github.com/christophschuhmann/improved-aesthetic-predictor
    """

    def __init__(self, *, dtype, model_id, model_filename):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        try:
            cached_path = hf_hub_download(model_id, model_filename)
        except EntryNotFoundError:
            cached_path = os.path.join(model_id, model_filename)
        state_dict = torch.load(cached_path, map_location=torch.device("cpu"), weights_only=True)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

def aesthetic_scorer(hub_model_id, model_filename):
    scorer = AestheticScorer(
        model_id=hub_model_id,
        model_filename=model_filename,
        dtype=torch.float32,
    )
    if is_torch_npu_available():
        scorer = scorer.npu()
    elif is_torch_xpu_available():
        scorer = scorer.xpu()
    else:
        scorer = scorer.cuda()

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def prompt_fn():
    return np.random.choice(animals), {}

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



class CustomResize():     
    # def __init__(self):
    #     self.size = 1   
    def __call__(self, img):
        w, h = img.size
        if w>=h:
          h=o2o_config.resolution
          w=int(w*o2o_config.resolution/h)
          img=F.resize(img,  (w, h))
          pad=int((w-o2o_config.resolution)/2)
          top=0
          left=pad
        else:
          w=o2o_config.resolution
          h=int(h*o2o_config.resolution/w)
          img=F.resize(img,  (w, h))
          pad=((h-o2o_config.resolution)/2)
          top=pad
          left=0
        return F.crop(img,top=top,left=left,height=o2o_config.resolution,width=o2o_config.resolution)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


def collate_fn(batch):
    return tuple(zip(*batch))
    

@dataclass
class ScriptArguments:
# "runwayml/stable-diffusion-v1-5"
# "stabilityai/stable-diffusion-2-1"
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

    print(f' ddo_config {o2o_config}' )
    print("-------------------------------------------------")
    print(f' args {args}' )

    fn_reward=aesthetic_scorer(args.hf_hub_aesthetic_model_id, args.hf_hub_aesthetic_model_filename)



    match o2o_config.dataset_index:
      case 0:
        dataset=ImageLionArtDatasetHugging(transform=transform,length=5000000,reward=100,vila_threshold=0.63)
      case 1:
        data_folder='./inputs/An_extremely_beautiful_Asian_girl_v1/'
        dataset=ImageScoreDataset(image_folder=data_folder,transform=transform,prompt="An extremely beautiful Asian girl")
      case 2:
        dataset=ImageArtPaintingDataset(image_folder='./inputs/selected-vincent-van-gogh',transform=transform,reward=0)
      case 3:
        dataset=ImagePickaPicDatasetHugging(image_folder="yuvalkirstain/pickapic_v2",transform=transform,reward=o2o_config.high_reward,length=5000000,vila_threshold=0.65,art_threshold=7.5,art_model=fn_reward)
      case 6:
        #   get only image with vila>5.5
          dataset=ImageLionArtDatasetHugging(transform=transform,length=20000,reward=100,vila_threshold=0.35)         
      case 7:
        data_folder='./inputs/cellphone_data'
        dataset = ImageScoreDataset(image_folder=data_folder,transform=transform,reward=o2o_config.high_reward,prompt="A man and woman using their cellphones, photograph")
      case 9:
        data_folder='./inputs/An_extremely_beautiful_Asian_girl_v1/'
        csv_file=data_folder+'data.csv'
        dataset=ImageScoreDatasetCSV(csv_file=csv_file,image_folder=data_folder)
      case 11:
        dataset= FilteredLaionArt(data_file="hoan17/test_csv_laion",transform=transform,reward=o2o_config.high_reward)
      case 12:
        dataset= SelectedPickaPic(transform=transform,reward=o2o_config.high_reward) 
    
      case _:
        dataset= SelectedPickaPic(transform=transform,reward=o2o_config.high_reward) 

    print("------------------------------------------------------------------------")
    print("Starting loading pipline -----------------------------------------------")

    
    test_mode=False
  

    if test_mode:
        dataset=ImagePickaPicDatasetHugging(image_folder="yuvalkirstain/pickapic_v2",transform=transform,reward=o2o_config.high_reward,length=10000,vila_threshold=0.60,art_threshold=7.0,art_model=fn_reward)        
        dataloader_train = DataLoader(dataset, batch_size=1)

        dataloader_train_iter=iter(dataloader_train)
        for epoch in range(50):
            next(dataloader_train_iter)
            if epoch%10==0:
                dataset.save_csv(f"./outputs/art_1100_6vila_7art_{epoch}p4.csv")
        
    else:
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
            fn_reward,
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
        name=f"dataset_index{o2o_config.dataset_index}_{model_note}_offbatch{off_batch}_e{num_epochs}"
        
        print("------------------------------------------------------------------------")
        print("Saving local    -----------------------------------------------")

        trainer.save_pretrained(f"./outputs/{model_note}")

        print("------------------------------------------------------------------------")
        print("Saving hub    -----------------------------------------------")


        if (args.hf_hub_model_id==""):
            print("Not load to github")
        else:
            trainer.push_to_hub(model_note)
