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
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser
import sys

from datasets import load_dataset
# sys.path.append('D:/THESIS/Thesis_Dev/trl0')
# sys.path.append('D:/THESIS/Thesis_Dev/trl0/codes')

print(sys.path)


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


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


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
        state_dict = torch.load(cached_path, map_location=torch.device("cpu"))
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

    print(f'score aesthetic: {scorer}.. \n')
    if is_npu_available():
        scorer = scorer.npu()
    elif is_xpu_available():
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

class ImageScoreDatasetHugging(Dataset):
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
        self.index=30
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

            if ("?resize") in img_link:
                img_link=img_link.split("?resize")[0]
            tail='jpg'
            file_name=f"./inputs/laion_saving/{self.index}_laion_image.{tail}"


            try:
                image = Image.open(requests.get(img_link,stream=True).raw).convert('RGB')
                image.save(file_name)
                image=image.convert('RGB')
                try_load=False
                # image = Image.open(file_name).convert('RGB')
            except:
                self.index+=1
                try_load=True



        if self.transform:
            image = self.transform(image)

        prompt =self.data['TEXT'][self.index]
        score=10*self.data['aesthetic'][self.index]
        batch=(image, torch.tensor(score, dtype=torch.float16), prompt,{},"NA")
        return batch

# class ImageScoreDatasetHuggingbk(Dataset):
#     def __init__(self,  data_file="laion/laion2B-multi-joined-translated-to-en",transform=None,length=5000):
#     #   https://huggingface.co/datasets/laion/laion2B-multi-joined-translated-to-en?row=74
#             # dataset= load_dataset("laion/laion-art")
#             # dataset.save_to_disk("./outputs/laion_art.hf")
#         save_name=data_file.split("/")[-1]
#         save_name=f"./outputs/{save_name}.hf"
#         try:   
#             print("loading dataset Laion local")
#             self.data= load_dataset(save_name)
#         except:            
#             print("loading dataset Laion cloud")
#             self.data= load_dataset(data_file)
#             print("saving dataset Laion")
            
#             self.data.save_to_disk(save_name)
#             # self.data.save_to_disk("./outputs/laion_art_en.hf")
        
#         self.data=self.data['train']
#         self.transform = transform
#         self.index=0
#         if length !=None:
#             self.data=self.data[:length]
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         self.index+=1
#         try_load=True
#         while try_load:
#             img_link=self.data['URL'][self.index]
#             print(f"ix  {self.index} link {img_link}")

#             if ("?resize") in img_link:
#                 img_link=img_link.split("?resize")[0]
#             tail='jpg'
#             file_name=f"./inputs/laion_saving/{self.index}_laion_image.{tail}"

#             if self.data['LANGUAGE'] !='en':
#                 pass
#             if self.data['WIDTH'] <300:
#                 pass

#             if self.data['HEIGHT'] <300:
#                 pass

#             if self.data['prediction']>4 and self.data['prediction']<5.4:
#                 rd=np.random.randint(100)
#                 if rd>=20:
#                     pass
            
                
#             try:
#                 image = Image.open(requests.get(img_link,stream=True).raw).convert('RGB')
#                 image.save(file_name)
#                 image=image.convert('RGB')
#                 try_load=False
#                 # image = Image.open(file_name).convert('RGB')
#             except:
#                 self.index+=1
#                 try_load=True



#         if self.transform:
#             image = self.transform(image)

#         prompt =self.data['TEXT'][self.index]
#         score=10*self.data['prediction'][self.index]
#         batch=(image, torch.tensor(score, dtype=torch.float16), prompt,{},"NA")
#         return batch



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


    # print("Batch of prompts:", prompts)
    # break  # Remove this to iterate over the entire dataset
def collate_fn(batch):
    return tuple(zip(*batch))
    


@dataclass
class ScriptArguments:
# "runwayml/stable-diffusion-v1-5"
    pretrained_model: str = field(
        default="stabilityai/stable-diffusion-2-1", metadata={"help": "the pretrained model to use"}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
   
    #  "ddpo-finetuned-stable-diffusion"
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
    args, ddpo_config = parser.parse_args_into_dataclasses()
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

    transform_valid = transforms.Compose(
        [
            transforms.Resize(ddpo_config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(ddpo_config.resolution),
            # transforms.RandomHorizontalFlip() if True else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )


    print(f' ddo_config {ddpo_config}' )
    print("-------------------------------------------------")

    print(args)


    csv_file=ddpo_config.data_folder+'data.csv'

    if ddpo_config.load_dataset_huggingface:
        dataset=ImageScoreDatasetHugging(transform=transform,length=5000)
    else:
        dataset = ImageScoreDataset(image_folder=ddpo_config.data_folder,transform=transform,reward=ddpo_config.high_reward)


    if ddpo_config.offpolicy_sample_batch_size>0:
        dataloader = DataLoader(dataset, batch_size=ddpo_config.offpolicy_sample_batch_size, shuffle=True)
    else:
        dataloader=None


    print("------------------------------------------------------------------------")
    print("Starting loading pipline -----------------------------------------------")


    pipeline = DefaultDDPOStableDiffusionPipeline(
        args.pretrained_model, pretrained_model_revision=args.pretrained_revision, use_lora=args.use_lora
    )
    if (args.load_folder!=''):
      pipeline.sd_pipeline.load_lora_weights(args.load_folder)


    trainer = DDPOTrainer(
        dataloader,
        ddpo_config,
        jpeg_compressibility(),
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


    trainer.save_pretrained(args.save_folder+str(ddpo_config.global_step+ddpo_config.num_epochs))



    # if (args.hf_hub_model_id==""):
    #     print("Not load to github")
    # else:
    #   trainer.push_to_hub(args.hf_hub_model_id+str(ddpo_config.global_step+ddpo_config.num_epochs))
