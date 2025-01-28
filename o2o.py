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
"""


!python examples/scripts/ddpo.py \
    --num_epochs=4 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=4 \
    --train_batch_size=2 \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb" \
    --logdir="ll" \
    --save_freq=200  \
    --save_folder="../ddpo_compressibility4"
"""

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
from codes.ddpo_config import DDPOConfig
from codes.ddpo_trainer import DDPOTrainer
from codes.modeling_sd_base import DefaultDDPOStableDiffusionPipeline
from codes.import_utils import is_npu_available, is_xpu_available

import io
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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


# list of example prompts to feed stable diffusion
animals = [
    "cat",
    "dog",
    "horse",
    "monkey",
    "rabbit",
    "zebra"
]

prompts = [
    "An extremely beautiful asian girl",
]

def prompt_fn():
    return np.random.choice(prompts), {}

def image_outputs_logger(image_data, global_step, accelerate_logger,caption='NA'):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
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


import math
from torchvision.transforms import functional as F

class CustomResize():     
    # def __init__(self):
    #     self.size = 1   
    def __call__(self, img):
        w, h = img.size
        if w>=h:
          h=ddpo_config.resolution
          w=int(w*ddpo_config.resolution/h)
          img=F.resize(img,  (w, h))
          pad=int((w-ddpo_config.resolution)/2)
          top=0
          left=pad
        else:
          w=ddpo_config.resolution
          h=int(h*ddpo_config.resolution/w)
          img=F.resize(img,  (w, h))
          pad=((h-ddpo_config.resolution)/2)
          top=pad
          left=0
        return F.crop(img,top=top,left=left,height=ddpo_config.resolution,width=ddpo_config.resolution)


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



class ImageScoreDatasetScore(Dataset):
    def __init__(self, csv_file, image_folder,transform=None,reward=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.reward=reward

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        # score = self.data.iloc[idx, 1]
        prompt = self.data.iloc[idx, 2]
        
        if self.reward==None:
            temp=str(self.data.iloc[idx, 0]).split("_")[0]
            score=int(temp)
        else:
            score=self.reward

        if self.transform:
            image = self.transform(image)
        batch=(image, torch.tensor(score, dtype=torch.float32), prompt,{},self.data.iloc[idx, 0])
        return batch

class ImageLionDatasetHugging(Dataset):
    def __init__(self,  data_file="Nguyen17/laion_art_en",transform=None,length=5000):
            # dataset= load_dataset("laion/laion-art")
            # dataset.save_to_disk("./outputs/laion_art.hf")
        try:   
            print("loading dataset Laion")
            self.data= load_dataset("./outputs/laion_art_en.hf")
        except:
            self.data= load_dataset(data_file)
            print("saving dataset Laion")
            save_name=data_file.split("/")[-1]
            self.data.save_to_disk(f"./outputs/{save_name}.hf")
        
        self.data=self.data['train']
        self.transform = transform
        self.index=0
        self.length=length
        if length !=None:
            self.data=self.data[:length]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.index+=1
        try_load=True
        while try_load:
            img_link=self.data['URL'][self.index]
            print(f"ix  {self.index} link {img_link}")
            tail='jpg'
            file_name=f"./inputs/laion_saving/{self.index}_laion_image.{tail}"
            try_load=False

            try:
                image = Image.open(requests.get(img_link,stream=True).raw).convert('RGB')
                # image.save(file_name)
                
            except:
                if self.index>self.length:
                  self.index=0
                else:
                  self.index+=1
                try_load=True

        if self.transform:
            image = self.transform(image)

        prompt =self.data['TEXT'][self.index]
        print(f"promt index{idx}_{self.index} {prompt}")
        score=10*self.data['aesthetic'][self.index]
        batch=(image, torch.tensor(score, dtype=torch.float16), prompt,{},"NA")
        return batch


class ImageScoreDataset(Dataset):
    def __init__(self,  image_folder,transform=None,reward=None,prompt="An extremely beautiful Asian girl"):
        self.data = []
        for file in os.listdir(image_folder):
          self.data.append(file)
        self.image_folder = image_folder
        self.transform = transform
        self.reward=reward
        self.prompt=prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data[idx])
        image = Image.open(img_name).convert('RGB')
        prompt = self.prompt
        score=self.reward

        if self.transform:
            image = self.transform(image)
        batch=(image, torch.tensor(score, dtype=torch.float16), prompt,{},self.data[idx])
        return batch


class ImageArtDataset(Dataset):
    def __init__(self,  image_folder,transform=None,reward=0):
        self.data = []
        for file in os.listdir(image_folder):
          nn=len(file)-4
          if file[nn:]==".jpg":
            self.data.append(file)
        self.image_folder = image_folder
        self.transform = transform
        self.reward=reward

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data[idx])
        image = Image.open(img_name)
        txt_file = str(img_name)+'.txt'
        # with open('readme.txt') as f:
        #   lines = f.readlines()
        prompt=pd.read_csv(txt_file)
        
        # prompt=prompt.iloc[0, 0]
       
       
        print(prompt)
        prompt="".join(list(prompt))
        print(prompt)

        if self.transform:
            image = self.transform(image)
        batch=(image, torch.tensor(self.reward, dtype=torch.float16), prompt,{},"")
        return batch

class ImagePickaPicDatasetHugging(Dataset):
    def __init__(self,  image_folder="yuvalkirstain/pickapic_v2",transform=None,reward=100,length=5000):

        self.dataset= load_dataset(image_folder,streaming=True)['train']
        self.it=iter(self.dataset)
        self.transform = transform
        self.index=0
        self.length=length
        self.score=reward
        self.item=None
        self.data={}
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        print(f"get item dataset pickapic {idx}")
        if idx in self.data.keys():
          return self.data[idx]
        try_load=True
        while try_load:
            self.item=next(self.it)
            print(f"in getitem {self.item['caption']}")
            img_name=None
            if int(self.item['label_0'])==1:
              img_name='jpg_0'
            if int(self.item['label_1'])==1:
              img_name='jpg_1'
            if img_name:
              try:
                image=Image.open(io.BytesIO(self.item[img_name])).convert("RGB") 
                try_load=False
              except:
                try_load=True              

        if self.transform:
            image = self.transform(image)

        prompt =self.item['caption']
        batch=(image, torch.tensor(self.score, dtype=torch.float16), prompt,{},"NA")
        self.data[idx]=batch    
        print(f"get item dataset pickapic {idx} caption {prompt}")

        return batch

def collate_fn(batch):
    return tuple(zip(*batch))
    


@dataclass
class ScriptArguments:
# "runwayml/stable-diffusion-v1-5"
# 
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "the pretrained model to use"}
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
    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    args, ddpo_config = parser.parse_args_into_dataclasses(return_remaining_strings=True)[:2]
    ddpo_config.project_kwargs = {
        "logging_dir": "./outputs/logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./outputs/",
    }



    transform = transforms.Compose(
        [
            transforms.Resize(ddpo_config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(ddpo_config.resolution),
            transforms.RandomHorizontalFlip() if True else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )

    print(f' ddo_config {ddpo_config}' )
    print("-------------------------------------------------")

    print(args)

#  data_folder: str = './inputs/An_extremely_beautiful_Asian_girl_v1'
# data_folder: str = './inputs/selected-vincent-van-gogh'

    match ddpo_config.dataset_index:
      case 0:
        dataset=ImageLionDatasetHugging(transform=transform,length=4000)
      case 1:
        # all score csv
        data_folder='./inputs/An_extremely_beautiful_Asian_girl_v1'
        csv_file=data_folder+'data.csv'
        dataset=ImageScoreDatasetScore(csv_file=csv_file, image_folder=data_folder,transform=transform,reward=None)
      case 2:
        dataset=ImageArtDataset(image_folder='./inputs/selected-vincent-van-gogh',transform=transform,reward=0)
      case 3:
        dataset=ImagePickaPicDatasetHugging(image_folder="yuvalkirstain/pickapic_v2",transform=transform,reward=ddpo_config.high_reward,length=5000)
      case 7:
        data_folder='./inputs/cellphone_data'
        dataset = ImageScoreDataset(image_folder=data_folder,transform=transform,reward=ddpo_config.high_reward,prompt="A man and woman using their cellphones, photograph")
      case _:
        dataset=ImageLionDatasetHugging(transform=transform,length=5000)



    # if ddpo_config.offpolicy_sample_batch_size>0:
    #     dataloader = DataLoader(dataset, batch_size=ddpo_config.offpolicy_sample_batch_size, shuffle=True)
    # else:
    #     dataloader=None


    print("------------------------------------------------------------------------")
    print("Starting loading pipline -----------------------------------------------")


    pipeline = DefaultDDPOStableDiffusionPipeline(
        args.pretrained_model, pretrained_model_revision=args.pretrained_revision, use_lora=args.use_lora
    )
    if (args.load_folder!=''):
      pipeline.sd_pipeline.load_lora_weights(args.load_folder)


    trainer = DDPOTrainer(
        dataset,
        ddpo_config,
        aesthetic_scorer(args.hf_hub_aesthetic_model_id, args.hf_hub_aesthetic_model_filename),
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )
    print("\n")
    print("------------------------------------------------------------------------")
    print("Starting training        -----------------------------------------------")


    epochs=ddpo_config.num_epochs
    trainer.train(epochs=epochs)

    print("\n")
    print("------------------------------------------------------------------------")
    print("Starting saving model    -----------------------------------------------")

    model_note=ddpo_config.huggingface_note
    num_epochs=ddpo_config.global_step+ddpo_config.num_epochs
    off_batch=ddpo_config.offpolicy_sample_batch_size
    name=f"dataset_index{ddpo_config.dataset_index}_{model_note}_offbatch{off_batch}_e{num_epochs}"
    
    trainer.save_pretrained(model_note)
    if (args.hf_hub_model_id==""):
        print("Not load to github")
    else:
      trainer.push_to_hub(model_note)
