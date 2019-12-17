# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 23:01:15 2019

@author: HongJea
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image 
import dataset
import torch 
import torch.nn.functional as F
import copy
import time
from torch import nn 
from torch.autograd import Variable
from torch.utils import data as da 
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import voc
import os 
import argparse
from optimizers import RAdam
from torchsummary import summary
import torchvision 
import torch.backends.cudnn as cudnn
from unet import Unet2D
from utils import optimize_linear
from losses import DiceLoss, SmoothCrossEntropyLoss
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, average_precision_score, jaccard_score
import torchvision.transforms as transforms


def loss_plot(train_loss, valid_loss, type_, task):
    
    '''
        Args:
            type_: base, ada, adt
            task: clf or seg
    '''
    
    train_max= np.max(train_loss)
    valid_max= np.max(valid_loss)
    
    ylim= np.median([train_max, valid_max, 6])+ 1
    
    plt.figure(figsize= (8, 6))
    plt.title('%s %s Loss'%(type_, task))
    plt.plot(train_loss, 'g', label= 'Train Loss')
    plt.plot(valid_loss, 'r', label= 'Val Loss')
    plt.ylim([0, ylim])
    plt.grid(True)
    plt.legend(loc= 'upper right')
    plt.savefig('./figure/%s_%s_loss.png'%(type_, task), dpi= 300)
    plt.show()
    
    
def segmentation_output_image(sample_list, logit, epoch, col_len= 4):
    
    num_sample= len(sample_list)
    row_len= num_sample// col_len+ 1
    
    pred= F.softmax(logit, dim= 1)
    pred= pred.argmax(dim= 1)
    
    cmap = plt.get_cmap('tab20')
    
    width, height= 4* col_len, 3* row_len
    fig= plt.figure(figsize= (width, height))
    
    ax_list= [plt.subplot(row_len, col_len, i) for i in range(1, num_sample+ 1)]
    
    for i, ax in enumerate(ax_list):
        ax.matshow(pred[i], cmap= cmap)
        
    fig.savefig('./figure/sample_%s_epoch.png', dpi= 300)
    fig.show()


def draw_plot(real_photo, segmentationmap, predict_map, epoch, model_name, i):
    from matplotlib import cm
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([248/256, 24/256, 148/256, 1])
    newcolors[:25, :] = pink
    cmap = ListedColormap(newcolors)

    color= ListedColormap([(c, c, c) for c in np.linspace(0, 1, 21)])
    
    fig= plt.figure(figsize= (21, 6))
    ax1, ax2, ax3= plt.subplot(131), plt.subplot(132), plt.subplot(133)
    
    ax1.imshow(np.asarray(real_photo))
    ax1.set_title('Real Image')
    ax1.tick_params(axis= 'both',
                    which= 'both',
                    labelbottom= False,
                    labelleft= False)

    ax2.matshow(segmentationmap.squeeze(), cmap= cmap)
    ax2.set_title('True Label')
    ax2.tick_params(axis= 'both',
                    which= 'both',
                    labeltop= False,
                    labelleft= False)
    
    ax3.matshow(predict_map.squeeze(), cmap= cmap)
    ax3.set_title('Model Prediction at %s epoch %s model'%(epoch, model_name))
    ax3.tick_params(axis= 'both',
                    which= 'both',
                    labeltop= False,
                    labelleft= False)
    
    fig.savefig('./figure/result_%s_eopch_%s_%s.png'%(epoch, model_name, i), dpi= 300)
    fig.show()


if __name__== '__main__':
    from tqdm import tqdm

    label_path = "seg_da/VOCdevkit/VOC2010/SegmentationClass/"
    image_path = "seg_da/VOCdevkit/VOC2010/JPEGImages"

    trainset = dataset.voc_seg(label_path, image_path, cut_out=False, smooth = False)

    total_idx = list(range(len(trainset)))
    split_idx = int(len(trainset) * 0.7)
    trn_idx = total_idx[:split_idx]
    val_idx = total_idx[split_idx:]

    model_path = ['/data1/workspace/geongyu/deep_lr_prj/segmentation-adv-smooth/49-model.pth', 
                '/data1/workspace/geongyu/deep_lr_prj/segmentation-none-smooth/49-model.pth', 
                '/data1/workspace/geongyu/deep_lr_prj/segmentation-none-cut-out/49-model.pth',
                '/data1/workspace/geongyu/deep_lr_prj/segmentation-adv-cut-out/49-model.pth',
                '/data1/workspace/geongyu/deep_lr_prj/segmentation-adv-none/49-model.pth',
                '/data1/workspace/geongyu/deep_lr_prj/seg_norm/model_99_future.pth']

    model_name = ["Segmentation_ADV_Smooth", 'Segmentation_Smoothing', 'Segmentation_CutOut', 'Segmentation_ADV_CutOut', 'Segmentation_ADV', "Segmentation Baseline"]

    for model, name in tqdm(zip(model_path, model_name)) :
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=False, sampler=SubsetRandomSampler(trn_idx))
        net = Unet2D((3, 256, 256), 1, 0.1, num_classes=21)
        st = torch.load(model)
        net = nn.DataParallel(net).cuda()
        net.load_state_dict(st)

        data = iter(trainloader).next()
        model_output = net(data[0])
        file_name = data[2]
        target = data[1] 

        for i in range(20) :
            im = file_name[i]
            pre = model_output[i].cpu()
            tar = target[i] 

            real_photo= Image.open(im, 'r')
            resize = transforms.Resize((256, 256))
            real_photo= resize(real_photo)

            predict_map= F.softmax(pre, dim= 0)
            predict_map= predict_map.argmax(dim= 0)
            #import ipdb; ipdb.set_trace()
            draw_plot(real_photo, tar, predict_map, epoch= 50, model_name = name, i=str(i))

    
'''       
 color= ListedColormap([(c, c, c) for c in np.linspace(0, 1, 20)])
        
        fig= plt.figure(figsize= (21, 6))
        ax1, ax2, ax3= plt.subplot(131), plt.subplot(132), plt.subplot(133)
        ax1.imshow(np.asarray(real_photo))
        ax1.set_title('Real Image')
        ax1.tick_params(axis= 'both',
                        which= 'both',
                        labelbottom= False,
                        labelleft= False)

        ax2.matshow(tar, cmap= color)
        ax2.set_title('True Label')
        ax2.tick_params(axis= 'both',
                        which= 'both',
                        labeltop= False,
                        labelleft= False)
        
        ax3.matshow(predict_map, cmap= color)
        ax3.set_title('Model Prediction')
        ax3.tick_params(axis= 'both',
                        which= 'both',
                        labeltop= False,
                        labelleft= False)
'''