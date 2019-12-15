import dataset
from tqdm import tqdm 
import torch 
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
from losses import DiceLoss
from medpy.metric import binary
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="segmentation", type=str, help="Task Type, For example Segmentation or Classification")
parser.add_argument("--optim", default="adam", type=str, help="Optimizers")
parser.add_argument("--loss-function", default="cross_entropy", type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument('--method', default="adv", type=str)
args = parser.parse_args()

def train(model, trn_loader, criterion, optimizer, epoch, mode="segmentation"):
    trn_loss = 0
    start_time = time.time()
    sum_iou = 0 
    sum_acc = 0 
    for i, (image, target) in enumerate(trn_loader) :
        model.train()
        x = image.cuda()
        y = target.cuda()
        
        y_pred = model(x)
        loss = criterion(y_pred, y.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if mode == "segmentation" : 
            #from PIL import Image 
            #palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
            #colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
            #colors = (colors % 255).numpy().astype("uint8")
            #r = Image.fromarray(y_pred[0].byte().cpu().numpy().astype("uint8").reshape(256, 256))
            #r.putpalette(colors)
            #r.save("test3.png") 
            pass

        elif mode == "classification" :
            acc = accuracy_score(target.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            sum_acc += acc 
            measure = acc
        
        trn_loss += (loss)
        end_time = time.time()
        print(" [Training] [{0}] [{1}/{2}] Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}] Measure = [{4:.3f}]".format(epoch, i, len(trn_loader), loss.item(), end_time-start_time))
        start_time = time.time()

    trn_loss = trn_loss/len(trn_loader)
    if mode == "segmentation" : 
        #total_iou = sum_iou / len(trn_loader)
        #total_measure = total_iou
        pass
    elif mode == "classification" :
        total_acc = sum_acc / len(trn_loader)
        total_measure = total_acc

    # 모델 저장기준 수정해야함 (14 이건규)
    if epoch == 25 or epoch == 50 or epoch == 75 or epoch == 99 : 
        torch.save(model.state_dict(), '{0}{1}_{2}.pth'.format("./", 'model', epoch))

    return trn_loss#, total_measure

def validate(model, val_loader, criterion, criterion_fn, optimizer, epoch, mode="segmentation"):
    model.eval()
    val_loss = 0 
    adv_losses = 0
    sum_iou = 0 ]
    adv_sum_iou = 0
    start_time = time.time()
    with torch.no_grad() :
        for i, (data, target) in enumerate(val_loader) :
            model.train()
            x = data.cuda()
            y = target.cuda()

            y_pred = model(x)
            loss = criterion(y_pred, y.long())
            val_loss += (loss)
            
            if args.method == 'adv':
                # calculate gradients of the inputs
                ## make copies of the inputs, the model, the loss function to prevent unexpected effect to the model
                x_clone = Variable(x.clone().detach(), requires_grad=True).cuda()
                model_fn = copy.deepcopy(model)
                loss_fn = criterion_fn(model_fn(x_clone), y)

                optimizer.zero_grad()
                loss_fn.backward()

                grads = x_clone.grad

                # calculate perturbations of the inputs
                scaled_perturbation = optimize_linear(grads, eps=0.25, norm=np.inf)
                # make adversarial samples
                adv_x = Variable(x_clone+scaled_perturbation, requries_grad=True).cuda()
                
                # put acv_x into the model
                adv_y_pred = model_fn(adv_x)
                adv_loss = criterion_fn(adv_y_pred, y.long().clone().detach())
                adv_losses += adv_loss.item()
                
                optimizer.zero_grad()
            
            if mode == "segmentation" : 
                #iou = binary.jc(target.cpu().numpy(), y_pred.detach().cpu().numpy())
                #sum_iou += iou 
                #measure = iou 
                pass
            elif mode == "classification" :
                acc = accuracy_score(target.cpu().numpy(), y_pred.detach().cpu().numpy())
                sum_acc += acc 
                measure = acc
                
                adv_acc = accuracy_score(target.cpu().numpy(), adv_y_pred.detach().cpu().numpy())
                adv_sum_acc += adv_acc
                
            end_time = time.time()
            print(" [Validation] [{0}] [{1}/{2}] Losses = [{3:.4f}] Adv. Losses = [{4:.4f}] Time(Seconds) = [{5:.2f}] Measure [{5:.3f}]".format(epoch, i, len(val_loader), loss.item(), adv_loss.item(), end_time-start_time))
            start_time = time.time()
    
    if mode == "segmentation" : 
        #total_iou = sum_iou / len(val_loader)
        #total_measure = total_iou
        pass
    elif mode == "classification" :
        total_acc = sum_acc / len(val_loader)
        total_measure = total_acc

    # write your codes here
    val_loss = val_loss / len(val_loader)
    adv_losses /= len(val_loader)

    return val_loss, adv_losses #, total_measure

def draw_plot(real_photo, segmentationmap, predict_map) :
    import matplotlib.pyplot as plt 
    import seaborn as sns 
    
def main():
    if args.mode == "segmentation" :
        label_path = "seg_da/VOCdevkit/VOC2010/SegmentationClass/"
        image_path = "seg_da/VOCdevkit/VOC2010/JPEGImages"
        trainset = dataset.voc_seg(label_path, image_path)
        total_idx = list(range(len(trainset)))
        split_idx = int(len(trainset) * 0.7)
        trn_idx = total_idx[:split_idx]
        val_idx = total_idx[split_idx:]

    elif args.mode == "Classification" :
        info_path = "./VOCdevkit/VOC2010/ImageSets/Main"
        image_path = "./VOCdevkit/VOC2010/JPEGImages"
        trainset = dataset.voc_cls(info_path, image_path)
    else : 
        raise NotImplementedError

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False, sampler=SubsetRandomSampler(trn_idx))
    testloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False, sampler=SubsetRandomSampler(val_idx))

    if args.mode == "segmentation" :
        net = Unet2D((3, 256, 256), 1, 0.1, num_classes=21)
    elif args.mode == "Classification" :
        net = torchvision.models.resnet50(pretrained=False, num_classes=20)
    else : 
        raise NotImplementedError

    if args.optim == 'sgd' :
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    elif args.optim == 'adam' :
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    elif args.optim == 'radam' :
        optimizer = RAdam(net.parameters(), lr = 0.0001)

    net = nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    if args.loss_function == "bce" :
        criterion = nn.BCEWithLogitsLoss().cuda()
        criterion_fn = nn.BCEWithLogitsLoss().cuda()
    elif args.loss_function == "dice" :
        criterion = DiceLoss().cuda()
        criterion_fn = DiceLoss().cuda()
    elif args.loss_function == "cross_entropy" :
        criterion = nn.CrossEntropyLoss().cuda()
        criterion_fn = nn.CrossEntropyLoss().cuda()
    else :
        raise NotImplementedError
    
    losses = []
    val_losses = []
    for epoch in range(args.epochs) : 
        train(net, trainloader, criterion, criterion_fn, optimizer, epoch)
        validate(net, testloader, criterion, epoch)

    return losses, val_losses


if __name__ == '__main__':
    tr_loss, val_loss = main()
    import ipdb; ipdb.set_trace()
    
