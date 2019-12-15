# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 23:01:15 2019

@author: HongJea
"""

import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np


def loss_plot(train_loss, valid_loss, args):
    
    if args.type== 'base': args.type= 'Baseline'
    elif args.type== 'ada': args.type= 'Adversarial Attack'
    elif args.type== 'adt': args.type= 'Adversarial Training'
    else: raise NotImplementedError
    
    train_max= np.max(train_loss)
    valid_max= np.max(valid_loss)
    
    ylim= np.median([train_max, valid_max, 6])+ 1
    
    plt.figure(figsize= (8, 6))
    plt.title('%s %s Loss'%(args.type, args.task))
    plt.plot(train_loss, 'g', label= 'Train Loss')
    plt.plot(valid_loss, 'r', label= 'Val Loss')
    plt.ylim([0, ylim])
    plt.grid(True)
    plt.legend(loc= 'upper right')


def main():
    
    parser= argparse.ArgumentParser()
    
    parser.add_argument('--type', type= str, default= 'ada', help= 'Adversarial type. \n\
                        ada: Adversarial-attack, adt: Adversarial training, base: Baseline')
    parser.add_argument('--task', type= str, default= 'seg', help= 'Classification(clf) or Segmentation(seg)')

    args = parser.parse_args()
    
    if args.task== 'clf': args.task= 'Classification'
    elif args.task== 'seg': args.task= 'Segmentation'

    train_loss= torch.load("./model_state_dict/%s_%s/train_loss.pkl"%(args.type, args.task), 
                       map_location=torch.device('cpu'))
    valid_loss = torch.load("./model_state_dict/%s_%s/validation_loss.pkl"%(args.type, args.task), 
                        map_location=torch.device('cpu'))
    
    train_loss= [float(loss) for loss in train_loss]
    valid_loss= [float(loss) for loss in valid_loss]

    loss_plot(train_loss, valid_loss, args)


if __name__== '__main__':
    
    main()
