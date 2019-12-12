import dataset
from tqdm import tqdm 
import torch 
import time
import string
from torch import nn 
from torch.utils import data as da 
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import voc
import os 
import argparse
from torchsummary import summary
import torchvision 

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="Segmentation", type=str, help="Task Type, For example Segmentation or Classification")
parser.add_argument("--optim", default="Adam", type=str, help="Optimizers")
args = parser.parse_args()

def train(model, trn_loader, device, criterion, optimizer, epoch):
    trn_loss = 0
    start_time = time.time()

    for i, (image, target) in enumerate(trn_loader) :
        model.train()
        x = image.cuda()
        y = target.cuda()
        
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trn_loss += (loss)
        end_time = time.time()
        print(" [Training] [{0}] [{1}/{2}] Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}]".format(epoch, i, len(trn_loader), loss.item(), end_time-start_time))
        start_time = time.time()

    trn_loss = trn_loss/len(trn_loader)

    if epoch == 50 : 
        torch.save(model.state_dict(), '{0}{1}_{2}.pth'.format("./", 'model', epoch))

    return trn_loss

def validate(model, val_loader, device, criterion, epoch):
    model.eval()
    val_loss = 0 
    start_time = time.time()
    with torch.no_grad() :
        for i, (data, target) in enumerate(val_loader) :
            model.train()
            x = data.cuda()
            y = target.cuda()

            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += (loss)
            end_time = time.time()
            print(" [Validation] [{0}] [{1}/{2}] Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}]".format(epoch, i, len(val_loader), loss.item(), end_time-start_time))
            start_time = time.time()

    # write your codes here
    val_loss = val_loss / len(val_loader)

    return val_loss

def draw_plot(real_photo, segmentationmap, predict_map) :
    import matplotlib.pyplot as plt 
    import seaborn as sns 
    
def main():
    if args.mode == "Segmentation" :
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor()
        ])
        trainset = torchvision.datasets.VOCSegmentation("./seg_da/", year='2010', image_set='train', 
                                                download=True, transform=transform, target_transform=transform)
        testset = torchvision.datasets.VOCSegmentation("./seg_da/", year='2010', image_set='val', 
                                               download=True, transform=transform, target_transform=transform)
    elif args.mode == "Classification" :
        pass
    else : 
        raise NotImplementedError

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    
    return losses, val_losses


if __name__ == '__main__':
    import ipdb; ipdb.set_trace()
    