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


def performance_plot(perform1, perform2, perform3, perform4, model_type, label_list):
        
    min_len= np.min([len(perform1), len(perform2), len(perform3), len(perform4)])
    
    perform1= perform1[:min_len]
    perform2= perform2[:min_len]
    perform3= perform3[:min_len]
    perform4= perform4[:min_len]
         
    plt.figure(figsize= (12, 6))
    ax1, ax2= plt.subplot(121), plt.subplot(122)
    
    ax1.set_title('Base-cutout segmentation loss')
    ax1.plot(perform1, 'g', label= label_list[0])
    ax1.plot(perform2, 'r', label= label_list[1])
    ax1.set_ylim([0, 3])
    ax1.legend(loc= 'upper right')
    ax1.grid(True)
    
    ax2.set_title('Base-cutout segmentation mAP')
    ax2.plot(perform3, 'g', label= label_list[2])
    ax2.plot(perform4, 'r', label= label_list[3])
    ax2.set_ylim([0, .2])
    ax2.legend(loc= 'upper left')
    ax2.grid(True)
    
    plt.savefig('./model_state_dict/base_seg_cutout/loss.png', dpi= 300)
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
    
    fig.savefig('./figure/compare_result_%s_eopch.png'%epoch, dpi= 300)
    fig.show()


if __name__== '__main__':
        
    perform1= torch.load("./model_state_dict/base_seg_cutout/trainloss.pkl", 
                         map_location=torch.device('cpu'))
    perform2= torch.load("./model_state_dict/base_seg_cutout/validloss.pkl", 
                         map_location=torch.device('cpu'))
    perform3= torch.load("./model_state_dict/base_seg_cutout/trainacc.pkl", 
                         map_location=torch.device('cpu'))
    perform3= [j.mean() for j in perform3]
    
    perform4= torch.load("./model_state_dict/base_seg_cutout/validacc.pkl", 
                         map_location=torch.device('cpu'))
    perform4= [j.mean()/169 for j in perform4]
    
    label_list= ['train loss', 'valid loss', 'train jac', 'valid jac']
    
    min_len= np.min([len(perform1), len(perform2), len(perform3), len(perform4)])
    
    perform1= perform1[:min_len]
    perform2= perform2[:min_len]
    perform3= perform3[:min_len]
    perform4= perform4[:min_len]
         
    plt.figure(figsize= (12, 6))
    ax1, ax2= plt.subplot(121), plt.subplot(122)
    
    ax1.set_title('Base-cutout segmentation loss')
    ax1.plot(perform1, 'g', label= label_list[0])
    ax1.plot(perform2, 'r', label= label_list[1])
    ax1.set_ylim([0, 3])
    ax1.legend(loc= 'upper right')
    ax1.grid(True)
    
    ax2.set_title('Base-cutout segmentation mAP')
    ax2.plot(perform3, 'g', label= label_list[2])
    ax2.plot(perform4, 'r', label= label_list[3])
    ax2.set_ylim([0, .2])
    ax2.legend(loc= 'upper left')
    ax2.grid(True)
    
    plt.savefig('./model_state_dict/base_seg_cutout/loss.png', dpi= 300)
    plt.show()    
