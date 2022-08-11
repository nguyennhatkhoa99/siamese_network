from copyreg import pickle
import random
from PIL import Image
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import PIL.ImageOps
import pickle as pkl
from configuration import Config
import torch
import os
import glob
from pathlib import Path

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        
    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 

        if should_get_same_class: 
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                # print(img0_tuple[1], img1_tuple[1])
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break
        img0_path = os.path.join(Config.training_dir, img0_tuple[0])
        img1_path = os.path.join(Config.training_dir, img1_tuple[0])
        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)
        # img0 = np.load(img0_tuple[0])
        # img1 = np.load(img1_tuple[0])
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")
        
        # if self.should_invert:
        #     img0 = PIL.ImageOps.invert(img0)    
        #     img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


def train_split_test(dataset, val_split = 0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)

class TripletLossDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        
    def getfile(self, anchor, index):
        anchor_full_path = self.imageFolderDataset.imgs[index][0]
        if (anchor[0].split('/')[-2] == 'positive'):
            negative_folder_path = os.path.join(os.path.split(os.path.split(anchor[0])[0])[0], 'negative')
            negative_image_path = random.choice(list(Path(negative_folder_path).glob('*.jpg')))
            negative_full_path = os.path.join(negative_folder_path, negative_image_path)
            positive_image_path = random.choice(list(Path(os.path.split(anchor[0])[0]).glob('*.jpg')))
            positive_full_path = os.path.join(os.path.split(anchor[0])[0], positive_image_path)
            # print('hiihihih', anchor_full_path, positive_full_path, negative_full_path)
            should_get_same_class = random.randint(0, 1)
            if should_get_same_class:
                random_id = random.choice(os.listdir(Config.training_dir)).split('/')[-1]
                negative_folder_from_another_class = os.path.join(os.path.join(Config.training_dir, random_id),'positive')
                negative_image_from_another_class = random.choice(list(Path(negative_folder_from_another_class).glob('*.jpg')))
                negative_full_path = os.path.join(negative_folder_from_another_class, negative_image_from_another_class)
                # print('class',anchor_full_path, positive_full_path, negative_full_path)
                return anchor_full_path, positive_full_path, negative_full_path
            return anchor_full_path, positive_full_path, negative_full_path
        
        elif (anchor[0].split('/')[-2] == 'negative'):
            positive_folder_path = os.path.join(os.path.split(os.path.split(anchor[0])[0])[0], 'positive')
            positive_image_path = random.choice(list(Path(positive_folder_path).glob('*.jpg')))
            positive_full_path = os.path.join(positive_folder_path, positive_image_path)
            anchor_image_path = random.choice(list(Path(positive_folder_path).glob('*.jpg')))
            anchor_full_path = os.path.join(positive_folder_path, anchor_image_path)
            # print('negative hiihihih', anchor_full_path,  positive_image_path, anchor[0])
            return positive_full_path,  anchor_full_path, anchor[0]

    def __getitem__(self, index):
        anchor = self.imageFolderDataset.imgs[index]
        anchor_full_path, positive_full_path, negative_full_path = self.getfile(anchor, index)
        # print(index, 'anchor: ', anchor_full_path, 'negative: ' ,negative_full_path,'positive: ' ,positive_full_path)
        anchor_img = Image.open(anchor_full_path)
        postive_img = Image.open(positive_full_path)
        negative_img = Image.open(negative_full_path)
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            postive_img = self.transform(postive_img)
            negative_img = self.transform(negative_img)
        return anchor_img, postive_img, negative_img
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)