from PIL import Image
from datasets import load_dataset
import torch
import requests
import os
import pandas as pd

from torch.utils.data import Dataset


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
        log_note=None
        while try_load:
            if self.data['WIDTH'][self.index]<700:
                log_note='small size'
                try_load=True
            elif self.data['HEIGHT'][self.index]<700:
                log_note='small size'
                try_load=True
            elif self.data['HEIGHT'][self.index]>self.data['WIDTH'][self.index]*1.2:
                try_load=True
                log_note='non square size'
            elif self.data['WIDTH'][self.index]>self.data['HEIGHT'][self.index]*1.2:
                try_load=True
                log_note='non square size'

            else:
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
                    log_note='max length index'

        if log_note:
            print(f"ImageLionDatasetHugging log {log_note} \n")

        if self.transform:
            image = self.transform(image)

        prompt =self.data['TEXT'][self.index]
        print(f"promt index{idx}_{self.index} {prompt}")
        score=10*self.data['aesthetic'][self.index]
        batch=(image, torch.tensor(score, dtype=torch.float16), prompt,{},"NA")
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
