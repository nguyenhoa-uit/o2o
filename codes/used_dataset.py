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


class SelectedPickaPic(Dataset):
    def __init__(self,  image_folder="./inputs/pick1100/", csv_file="./inputs/pick1100/pick1100.csv",transform=None,reward=100):
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


class FilteredLaionArt(Dataset):
    def __init__(self,  data_file="hoan17/test_csv_laion",transform=None,reward=100):
        super().__init__()
        save_name=data_file.split("/")[-1]
        local_save=f"./outputs/{save_name}.hf"

        # try:   
        #     print(f"loading dataset local file Laion {save_name}")
        #     self.data= load_dataset(local_save)
        # except:
        #     self.data= load_dataset(data_file)
        #     print("saving dataset Laion")
        #     save_name=data_file.split("/")[-1]
        #     self.data.save_to_disk(local_save)
        self.data= load_dataset(data_file)
      

        self.data=self.data['train']
        self.transform = transform
        self.reward=reward
        
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
    

class ImageLionArtDatasetHugging(Dataset):
    def __init__(self,  data_file="Nguyen17/laion_art_en",transform=None,length=5000,reward=None,vila_threshold=0.0):
        super().__init__()

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
        self.vila_threshold=vila_threshold
        self.reward=reward
        self.list_csv=[]
        if self.vila_threshold>0.0:
            self.vila_model=hub.load('https://tfhub.dev/google/vila/image/1')
        else:
            self.vila_model=None
        if length !=None:
            self.data=self.data[:length]
        self.count_im=0
    def __len__(self):
        return self.length
    def save_csv(self,filename):
        print(self.list_csv)

        prompt=[item["prompt"] for item in self.list_csv]
        vila_r=[item["vila_r"] for item in self.list_csv]
        link=[item["link"] for item in self.list_csv]

        # dictionary of lists
        dict = {'prompt': prompt, 'link': link}          
        df = pd.DataFrame(dict)
        df.to_csv(filename)

    def __getitem__(self, idx):
        try_load=True
        log_note=None
        img_link=""
        vila_r=0.0
        while try_load:
            self.index+=1
            if self.index>62470 and self.index<62479:
                 try_load=True
            elif self.index>96492 and self.index<96498:
                try_load=True
            elif not check_size_image(self.data['HEIGHT'][self.index],self.data['WIDTH'][self.index]):
                try_load=True
                log_note='Check size fails'
            else:
                img_link=self.data['URL'][self.index]
                tail='jpg'
                file_name=f"./inputs/laion_saving/{self.index}_laion_image.{tail}"
                try_load=False
                log_note=f"ix  {self.index} link {img_link}"
                try:
                    image = Image.open(requests.get(img_link,stream=True).raw).convert("RGB") 
                    if self.transform:
                        image = self.transform(image)
                    if self.vila_threshold>0.0:
                        vila_r=predict_vila_image(F.to_pil_image(image, mode=None),self.vila_model)
                    else:
                        vila_r=10
                    
                    if vila_r<self.vila_threshold:
                        try_load=True
                        log_note=f'low vila r: {vila_r}'


                    # image.save(file_name)                
                except:
                    if self.index>self.length:
                        self.index=0
                    else:
                        self.index+=1
                    try_load=True
                    log_note='max length index'
            if log_note:
                print(f"ImageLionDatasetHugging log {log_note} index {self.index} imgs {self.count_im} \n")


        prompt =self.data['TEXT'][self.index]
        print(f"promt index{idx}_{self.index} {prompt}, vila {vila_r}")

        # if reward=None, set reward= aesthetic in dataset
        if self.reward:
            reward=self.reward
        else:
            reward=10*self.data['aesthetic'][self.index]
        batch=(image, torch.tensor(reward, dtype=torch.float16), prompt,{},"NA")
        self.list_csv.append({"prompt":prompt,"link":img_link,"vila_r":float(vila_r)})
        self.count_im+=1
        return batch

class ImageScoreDatasetCSV(Dataset):
    def __init__(self, csv_file, image_folder,transform=None,reward=100):
        super().__init__()
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

class ImageArtPaintingDataset(Dataset):
    def __init__(self,  image_folder,transform=None,reward=0):
        super().__init__()
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
        
        return batch


def check_size_image1(height,width): 
    if not height:
        return False
    if not width:
        return False
    if height<400:
        return False
    if width<400:
        return False  
    if float(height/width)>1.3:
        return False
    if float(width/height)>1.3:
        return False 
    return True
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
