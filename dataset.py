# import some packages you need here
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import string 
from torchvision import transforms
import cv2

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def read_all(path):
    dataest = []
    files = os.listdir(path)
    for fi in files:
        if('trainval' not in fi):
            num = long(1)
            for i,str in enumerate(object_categories,long(1)):
                if (str in fi):
                    num = i
                    break
            if('train' in fi):
                f = open(path+"/"+fi)
                iter_f = iter(f)
                for line in iter_f:
                    line = line[0:11]
                    dataest.append([line,num])
            else:
                f = open(path+"/"+fi)
                iter_f = iter(f)
                for line in iter_f:
                    line = line[0:11]
                    dataest.append([line,num])
    return dataest

class voc_cls(Dataset):
    def __init__(self, infor_path, input_path) :
        self.info_path = infor_path
        self.path = input_path
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.datasets = read_all(self.info_path)
        self.transform_1 = transforms.ToTensor()
        self.transform_2 = transforms.Normalize(0.5, 0.5)
    
    def __len__(self) :
        return len(self.datasets)
    
    def __getitem__(self, idx) :
        img = self.datasets[idx]
        img_path = self.path 
        imgs = path + img[0] + ".jpg"

        image = cv2.imread(imgs)
        image = cv2.resize(image, (224, 224))
        image = self.transform_1(image)

        label = torch.tensor(img[1])
        
        return image, label 
    
    def get_classes(self) :
        return self.classes