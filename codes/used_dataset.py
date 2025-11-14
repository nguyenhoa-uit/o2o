from PIL import Image
from datasets import load_dataset
import torch
import requests
import os
import pandas as pd
import io
from torch.utils.data import Dataset
import tensorflow_hub as hub
from torchvision.transforms import functional as F
from codes.utils import predict_vila_image

class FilteredLaionAesthetic(Dataset):
    def __init__(self,  dataset_link="hoan17/laion_en_ae_filtered",transform=None):
        super().__init__()
        save_name=dataset_link.split("/")[-1]
        local_save=f"./outputs/{save_name}.hf"
        self.data= load_dataset(dataset_link)
        self.data=self.data['train']
        self.transform = transform
        self.reward=1
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      
        img_link=self.data['URL'][idx]
        try:
            image = Image.open(requests.get(img_link,stream=True).raw).convert("RGB") 
            if self.transform:
                image = self.transform(image)               
        except Exception as err:
            print(f'Error load image: {idx} ----------------- {err}')
        prompt =self.data['TEXT'][idx]
        # if reward=None, set reward= aesthetic in dataset
        batch=(image, torch.tensor(self.reward, dtype=torch.float16), prompt,{},"NA")
        return batch

class FilteredLaionArt(Dataset):
    def __init__(self,  dataset_link="hoan17/test_csv_laion",transform=None):
        super().__init__()
        save_name=dataset_link.split("/")[-1]
        local_save=f"./outputs/{save_name}.hf"
        self.data= load_dataset(dataset_link)
        self.data=self.data['train']
        self.transform = transform
        self.reward=1
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      
        img_link=self.data['link'][idx]
        try:
            image = Image.open(requests.get(img_link,stream=True).raw).convert("RGB") 
            if self.transform:
                image = self.transform(image)               
        except Exception as err:
            print(f'Error load image: {err}')
        prompt =self.data['prompt'][idx]
        # if reward=None, set reward= aesthetic in dataset
        batch=(image, torch.tensor(self.reward, dtype=torch.float16), prompt,{},"NA")
        return batch



class SelectedPickaPic(Dataset):
    def __init__(self,  image_folder="./inputs/pick1050/", csv_file="./inputs/pick1050/pick1050.csv",transform=None,reward=100):
        super().__init__()

        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.reward=reward 


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data["good_jpg"][idx])
        image = Image.open(img_name).convert('RGB')
        prompt = self.data["prompt"][idx]
        score=self.reward

        if self.transform:
            image = self.transform(image)
        batch=(image, torch.tensor(score, dtype=torch.float16), prompt,{},"NA")
        print(f"getitem_dataset: idx= {idx}")
        return batch



class VuvuzelaSet(Dataset):
    def __init__(self,  image_folder="./inputs/vuvuzela/images", csv_file="./inputs/vuvuzela/vuvuzela.csv",transform=None,reward=100):
        super().__init__()

        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.reward=reward 


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, str(self.data["number"][idx])+".jpg")
        image = Image.open(img_name).convert('RGB')
        prompt = self.data["prompt"][idx]
        score=self.reward

        if self.transform:
            image = self.transform(image)
        batch=(image, torch.tensor(score, dtype=torch.float16), prompt,{},"NA")
        print(f"getitem_dataset: idx= {idx}")
        return batch


class ImageScoreDataset(Dataset):
    def __init__(self,  image_folder,transform=None,reward=100,prompt="An extremely beautiful Asian girl"):
        super().__init__()
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
        print(f"item idx {idx}")
        return batch

# used to filter images
class ImagePickaPicDatasetHugging(Dataset):
    def __init__(self,  image_folder="yuvalkirstain/pickapic_v2",transform=None, \
                 reward=100,length=5000,vila_threshold=0.0, save_mode=False):
        super().__init__()

        self.dataset= load_dataset(image_folder,streaming=True)['train']
        self.it=iter(self.dataset)
        self.transform = transform
        self.list_csv=[]

        self.length=length
        self.reward=reward
        # self.data={}
        self.vila_threshold=vila_threshold
        if self.vila_threshold>0.0:
            self.vila_model=hub.load('https://tfhub.dev/google/vila/image/1')
        else:
            self.vila_model=None
        self.count=0
        self.index=-1
    def __len__(self):
        return self.length
    def save_csv(self,filename):
       
        prompts=[item["prompt"] for item in self.list_csv]
        good_jpgs=[item["good_jpg"] for item in self.list_csv]
        bad_jpgs=[item["bad_jpg"] for item in self.list_csv]
        origin_indexs=[item["origin_index"] for item in self.list_csv]

        # dictionary of lists
        dict = {'prompt': prompts, 'good_jpg': good_jpgs,'bad_jpg':bad_jpgs,"origin_index":origin_indexs}          
        df = pd.DataFrame(dict)
        df.to_csv(filename)

    def __getitem__(self, idx):
        # if idx in self.data.keys():
        #   return self.data[idx]
        try_load=True
        log_note=""
        bad_image_name=None

        while try_load:
            img_name=None
            self.index+=1
            item=next(self.it)


            
            if self.index<1:
                print(f'next {self.index}')
                continue
            prompt =item['caption']

            if int(item['label_0'])==1:
              img_name='jpg_0'
              bad_image_name='jpg_1'
            if int(item['label_1'])==1:
              img_name='jpg_1'
              bad_image_name='jpg_0'

            if img_name:
              try:
                image_good=Image.open(io.BytesIO(item[img_name])).convert("RGB") 
                image_bad=Image.open(io.BytesIO(item[bad_image_name])).convert("RGB") 

                if not check_size_image(*image_good.size):
                    log_note='Check size fails'
                elif self.transform:
                    image = self.transform(image_good)
                    if self.vila_threshold>0.0:
                        vila_r=predict_vila_image(F.to_pil_image(image, mode=None),self.vila_model)
                    else:
                        vila_r=100
                    
                    if vila_r<self.vila_threshold:
                        log_note=f'low vila r: {vila_r}'
                    else:
              
                        try_load=False
                        log_note=f'Okie vila r: {vila_r}'
              except Exception as er:
                log_note=f'except something: er {er}'
            else:
                log_note="Only 0.5 in the sample"
                try_load=True 
            print(f"{try_load} pickapic - {log_note} caption {prompt} count{self.count}")           
        good_img_filename=f"good_{self.count}.jpg"
        bad_img_filename=f"bad_{self.count}.jpg"

        image_good.save(f"./outputs/pick1000/{good_img_filename}")
        image_bad.save(f"./outputs/pick1000/{bad_img_filename}")

        csv_row = {"prompt":prompt,"bad_jpg":bad_img_filename, "good_jpg":good_img_filename,"origin_index":self.index}
        self.list_csv.append(csv_row)

        self.count+=1
        batch=(image, torch.tensor(self.reward, dtype=torch.float16), prompt,{},"NA")
        # self.data[idx]=batch    
   
def check_size_image(height,width): 
    if not height:
        return False
    if not width:
        return False
    if height<500:
        return False
    if width<500:
        return False  
    if float(height/width)>1.25:
        return False
    if float(width/height)>1.25:
        return False 
    return True
