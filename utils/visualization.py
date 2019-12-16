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


def draw_plot(real_photo, segmentationmap, predict_map, epoch):

    color= ListedColormap([(c, c, c) for c in np.linspace(0, 1, 20)])
    
    fig= plt.figure(figsize= (21, 6))
    ax1, ax2, ax3= plt.subplot(131), plt.subplot(132), plt.subplot(133)
    
    ax1.imshow(np.asarray(real_photo))
    ax1.set_title('Real Image')
    ax1.tick_params(axis= 'both',
                    which= 'both',
                    labelbottom= False,
                    labelleft= False)

    ax2.matshow(segmentationmap.squeeze(), cmap= color)
    ax2.set_title('True Label')
    ax2.tick_params(axis= 'both',
                    which= 'both',
                    labeltop= False,
                    labelleft= False)
    
    ax3.matshow(predict_map.squeeze(), cmap= color)
    ax3.set_title('Model Prediction at %s epoch'%epoch)
    ax3.tick_params(axis= 'both',
                    which= 'both',
                    labeltop= False,
                    labelleft= False)
    
    fig.savefig('./figure/result_%s_eopch.png'%epoch, dpi= 300)
    fig.show()


if __name__== '__main__':
    
    real_photo= Image.open('./model_state_dict/logit_for_visualization/real_image.jpg', 'r')
    resize = transforms.Resize((256, 256))
    real_photo= resize(real_photo)
    
    segmentationmap= torch.load("./model_state_dict/logit_for_visualization/target.pkl", 
                                map_location=torch.device('cpu'))
    
    logit= torch.load("./model_state_dict/logit_for_visualization/y_pred.pkl", 
                      map_location=torch.device('cpu'))
    predict_map= F.softmax(logit, dim= 1)
    predict_map= predict_map.argmax(dim= 1)
    
    draw_plot(real_photo, segmentationmap, predict_map, epoch= 0)
    
    
    color= ListedColormap([(c, c, c) for c in np.linspace(0, 1, 20)])
    
    fig= plt.figure(figsize= (21, 6))
    ax1, ax2, ax3= plt.subplot(131), plt.subplot(132), plt.subplot(133)
    ax1.imshow(np.asarray(real_photo))
    ax1.set_title('Real Image')
    ax1.tick_params(axis= 'both',
                    which= 'both',
                    labelbottom= False,
                    labelleft= False)

    ax2.matshow(segmentationmap, cmap= color)
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
