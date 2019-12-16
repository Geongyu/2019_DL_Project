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
    def __init__(self, label_path, image_path, cut_out=False, smooth=False) :
        self.label_path = label_path
        self.image_path = image_path
        self.classes = ["background", 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.data_list = os.listdir(label_path)
        self.transform_1 = transforms.ToTensor()
        self.resize = transforms.Resize((256, 256))
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.cut_out = cut_out
    
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
        if self.cut_out == True :
            ct = cutout(mask_size = (32, 32, 1), p = 0.5, cutout_inside = True)
            image = ct(image)
        image = self.transform_1(image)
        image = self.normalize(image)

        label = Image.open(label)
        label = self.resize(label)
        label = np.array(label)
        label[label == 255] = 0
        uni = np.unique(label)
        
        case = torch.zeros(20)
        for i in uni :
            if i == 0 :
                pass
            else :
                case[i-1] = 1
        
        if self.smooth == True :
            case = case - 0.01
            case = np.abs(case)

        

        return image, case 
    
    def get_classes(self) :
        return self.classes

class voc_seg(Dataset):
    def __init__(self, label_path, image_path, cut_out=True, smooth=False) :
        self.label_path = label_path
        self.image_path = image_path
        self.classes = ["background", 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.data_list = os.listdir(label_path)
        self.cut_out = cut_out
        self.smooth = smooth 
        self.transform_1 = transforms.ToTensor()
        self.resize = transforms.Resize((256, 256))
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    def __len__(self) :
        return len(self.data_list)
    
    def __getitem__(self, idx) :

        base = self.data_list[idx]
        label = os.path.join(self.label_path, base)
        image = os.path.join(self.image_path, base.replace(".png", ".jpg"))

        image = Image.open(image)
        image = self.resize(image)

        if self.cut_out == True :
            cut = cutout(mask_size = 32, p = 0.5, cutout_inside = True)
            image = cut(image)

        image = self.transform_1(image)
        image = self.normalize(image)

        label = Image.open(label)
        label = self.resize(label)
        label = np.array(label)

        if self.smooth == True :
            label = label - 0.01
            label = np.abs(label)

        label[label == 255] = 0

        label = torch.tensor(label)

        return image, label 
    
    def get_classes(self) :
        return self.classes



def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0
    
    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image
        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout