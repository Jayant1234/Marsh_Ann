
import numpy as np
import cv2
import csv

import torch
from torch.utils.data import Dataset
from PIL import Image

class MarshPlant_Dataset_pa(Dataset):
    def __init__(self, infile,transform=None):
        #initialize dataset
        self.imgfiles = []
        self.anns = []
        self.transform= transform
        with open(infile,'r') as f:
            reader = csv.reader(f,delimiter='\t')
            for row in reader:
                #print(row[0])
                self.imgfiles.append(row[0])
                ann = list(map(int,row[1:8]))
                self.anns.append(ann)

    def __len__(self):
        #return length of dataset
        return len(self.imgfiles)

    def __getitem__(self, idx):
        #return dataset[idx]
        #print(self.imgfiles[idx])
        y = torch.tensor(self.anns[idx]).float()  #annotations
        im = Image.open(self.imgfiles[idx])
        #im = cv2.imread(self.imgfiles[idx])
        #dst = cv2.fastNlMeansDenoisingColored(im,None,10,10,7,21)
        #im = cv2.resize(im, (512,512)) #resize patch
        #mean_value= im.mean()
        #subtracting each pixel of the image from mean
        #im= mean_value - im
        #im = np.transpose(im,(2,0,1))/ 255.
        #im = np.expand_dims(im, axis=0)
        #denoising instead of image mean centering
        #x = torch.from_numpy(im).float()
        x = self.transform(im)
        return {'X': x, 'Y': y}


class MarshPlant_Dataset_pc(Dataset):
    def __init__(self, infile,transform=None):
        #initialize dataset
        self.imgfiles = []
        self.anns = []
        self.transform= transform
        with open(infile,'r') as f:
            reader = csv.reader(f,delimiter='\t')
            for row in reader:
                #print(row[0])
                self.imgfiles.append(row[0])
                ann = list(map(int,row[1:10])) # change to 1:8 for pa and 1:10 for percent cover
                self.anns.append(ann)

    def __len__(self):
        #return length of dataset
        return len(self.imgfiles)

    def __getitem__(self, idx):
        #return dataset[idx]
        #print(self.imgfiles[idx])
        y = torch.tensor(self.anns[idx]).float()  #annotations
        im = Image.open(self.imgfiles[idx])
        #im = cv2.imread(self.imgfiles[idx])
        #dst = cv2.fastNlMeansDenoisingColored(im,None,10,10,7,21)
        #im = cv2.resize(im, (512,512)) #resize patch
        #mean_value= im.mean()
        #subtracting each pixel of the image from mean
        #im= mean_value - im
        #im = np.transpose(im,(2,0,1))/ 255.
        #im = np.expand_dims(im, axis=0)
        #denoising instead of image mean centering
        #x = torch.from_numpy(im).float()
        x = self.transform(im)
        return {'X': x, 'Y': y}
