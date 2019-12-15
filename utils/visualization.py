# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 23:01:15 2019

@author: HongJea
"""

import torch
import matplotlib.pyplot as plt
import argparse


def loss_plot(train_loss, valid_loss, args):
    
    if args.type== 'base': args.type= 'Baseline'
    elif args.type== 'ada': args.type= 'Adversarial Attack'
    elif args.type== 'adt': args.type= 'Adversarial Training'
    else: raise NotImplementedError
    
    plt.figure(figsize= (8, 6))
    plt.title('%s %s Loss'%(args.type, args.task))
    plt.plot(train_loss, 'g', label= 'Train Loss')
    plt.plot(valid_loss, 'r', label= 'Val Loss')
    plt.ylim([0, 6])
    plt.grid(True)
    plt.legend(loc= 'upper right')


def main():
    
    parser= argparse.ArgumentParser()
    
    parser.add_argument('--type', type= str, default= 'base', help= 'Adversarial type. \n\
                        ada: Adversarial-attack, adt: Adversarial training, base: Baseline')
    parser.add_argument('--task', type= str, default= 'clf', help= 'Classification(clf) or Segmentation(seg)')

    args = parser.parse_args()
    
    if args.task== 'clf': args.task= 'Classification'
    elif args.task== 'seg': args.task= 'Segmentation'

    train_loss= torch.load("./model_state_dict/%s_%s/train_loss.pkl"%(args.type, args.task), 
                       map_location=torch.device('cpu'))
    valid_loss = torch.load("./model_state_dict/%s_%s/validation_loss.pkl"%(args.type, args.task), 
                        map_location=torch.device('cpu'))

    loss_plot(train_loss, valid_loss, args)


