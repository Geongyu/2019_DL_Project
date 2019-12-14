# import some packages you need here
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import string 
from torchvision import transforms
import cv2
from PIL import Image 

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
        imgs = img_path + img[0] + ".jpg"

        image = cv2.imread(imgs)
        image = cv2.resize(image, (224, 224))
        image = self.transform_1(image)

        label = torch.tensor(img[1])

        return image, label 
    
    def get_classes(self) :
        return self.classes

class voc_seg(Dataset):
    def __init__(self, label_path, image_path) :
        self.label_path = label_path
        self.image_path = image_path
        self.classes = ["background", 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.data_list = os.listdir(label_path)
        self.transform_1 = transforms.ToTensor()
        self.resize = transforms.Resize((256, 256))
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    def __len__(self) :
        return len(self.data_list)
    
    def __getitem__(self, idx) :

        base = self.data_list[idx]
        label = os.path.join(self.label_path, base)
        image = os.path.join(self.image_path, base.replace(".png", ".jpg"))

        #label = self.label_path + "/" +  base
        #image = self.image_path + "/" +  base.replace("png", "jpg")
        
        image = Image.open(image)
        image = self.resize(image)
        image = self.transform_1(image)
        image = self.normalize(image)

        label = Image.open(label)
        label = self.resize(label)
        label = np.array(label)
        label[label == 255] = 0
        label = torch.tensor(label)

        return image, label 
    
    def get_classes(self) :
        return self.classes