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
from utils.optimizers import RAdam
from torchsummary import summary
import torchvision 
import torch.backends.cudnn as cudnn
from models.unet import Unet2D
from utils.losses import DiceLoss

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="Segmentation", type=str, help="Task Type, For example Segmentation or Classification")
parser.add_argument("--optim", default="Adam", type=str, help="Optimizers")
parser.add_argument("--loss-function", default="BCE", type=str)
parser.add_argument("--epochs", default=50, type=int)
args = parser.parse_args()

def train(model, trn_loader, criterion, optimizer, epoch):
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

def validate(model, val_loader, criterion, epoch):
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
        info_path = "./VOCdevkit/VOC2010/ImageSets/Main"
        image_path = "./VOCdevkit/VOC2010/JPEGImages"
        trainset = dataset.voc_cls(info_path, image_path)
    else : 
        raise NotImplementedError

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    if args.mode == "Segmentation" :
        net = torchvision.models.resnet50(pretrained=False, num_classes=20)
    elif args.mode == "Classification" :
        net = Unet2D((3, 224, 224), 1, 0.1)
    else : 
        raise NotImplementedError

    if args.optim == 'sgd' :
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    elif args.optim == 'adam' :
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    elif args.optim == 'radam' :
        optimizer = RAdam(net.parameters(), lr = 0.0001)

    net = nn.DataParallel(net).cuda()
    cudnn.benchmark = True
    if args.loss_function == "bce" :
        criterion = nn.BCELoss()
    elif args.loss_function == "dice" :
        criterion = DiceLoss().cuda()
    elif args.loss_function == "cross_entorpy" :
        criterion = nn.CrossEntropyLoss()
    else :
        raise NotImplementedError
    
    losses = []
    val_losses = []
    for epoch in range(args.epochs) : 
        train(net, trainloader, criterion, optimizer, epoch)
        validate(net, testloader, criterion, epoch)

    return losses, val_losses


if __name__ == '__main__':
    import ipdb; ipdb.set_trace()
    