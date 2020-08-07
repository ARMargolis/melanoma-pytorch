import os, sys
import pandas as pd 
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

from PIL import Image
from torchvision import transforms, utils


default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MelanomaDataset(data.Dataset):
    # TODO initialize conv layers and fc layers in a particular way. Important for our experiment
    # TODO add cuda support
    
    def __init__(self, imgs_dir, label_csv, train, device = default_device, **specifications):
        """
        Initializes the dataset. Will use .jpg images.
        img_dir (str): Absolute path to directory where the .jpg files are stored
        label_csv (str): Absolute path to csv that contains the labels and metadata for the images
        specifications (optional): any extra keyword args are stored in a dict. Can include:
            -transform
            -resolution: what size do we want to resize the images to? Square or rectangular?
        """
        
        #  Handle specifications
        if specifications:
            print('Specifications:')
            for key, value in specifications.items():
                print(key, ' : ', value)
            print('----------------------')
        #images generally are 1053x1872 coming in - will centercrop to 1000 x 1000
        self.initial_resolution = (1053,1872)
        if 'initial_resolution' in specifications:
            if isinstance(specifications['initial_resolution']):
                self.initial_resolution = specifications['initial_resolution']
            else:
                print('Invalid format for inital resolution. Give a tuple')
        # Set resolution of actual samples to feed into the model
        self.resolution = 244 # 1000 default

        # Figure out what transforms to use
        if 'transform' not in specifications:
            # If I didn't specify a transform
            if train:
                # if a training dataset
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(self.initial_resolution),
                        transforms.CenterCrop(self.initial_resolution[0]), #get a square image
                        transforms.RandomResizedCrop(size=self.resolution, scale=(0.8, 1.0)),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomHorizontalFlip(p=0.5),
                        #transforms.RandomRotation(degrees=10, resample = Image.BICUBIC, expand = True),
                        #transforms.CenterCrop(self.resolution),
                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) #Taken from overall dataset
                    ]
                )
            else:
                # if test
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(self.initial_resolution),
                        transforms.CenterCrop(self.initial_resolution[0]), #get a square image
                        transforms.CenterCrop(self.resolution), 
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                    ]
                )
        else:
            if specifications['transform'] == None:
                # If I specifically specify no transform, then the only transform is to_tensor
                self.transform = transforms.functional.to_tensor #self.transform is a function
            else:
                # If I directly feed in a transform, use that
                self.transform = specifications['transform']


        self.imgs_dir = imgs_dir
        # Get the label stuff
        # Columns: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target
        # target: 0 == benign, 1 == malignant
        self.label_df = pd.read_csv(label_csv)
    
    def __len__(self):
        return self.label_df.shape[0]

    def __getitem__(self, idx):
        """
            Get a single sample
            Want this method to be very fast, i.e. few or no if statements, etc.
        """
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()
        
        name = self.label_df.iloc[idx]['image_name']
        label = self.label_df.iloc[idx]['target']

        # get PIL image
        image = Image.open(
            os.path.join(self.imgs_dir,name+'.jpg')
        )

        image = self.transform(image)

        return (image, torch.LongTensor([label]) )
    
    def display_sample(self, idx):
        pic, label = self.__getitem__(idx)
        pic = transforms.functional.to_pil_image(pic)
        print('----------------------\nlabel: ', label.cpu().numpy())
        # if not using in jupyter notebook context, show the image directly instead of returning it
        # pic.show()
        return pic
