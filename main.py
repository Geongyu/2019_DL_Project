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

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="segmentation", type=str, help="Task Type, For example segmentation or classification")
parser.add_argument("--optim", default="radam", type=str, help="Optimizers")
parser.add_argument("--loss-function", default="bce", type=str)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument('--method', default="adv", type=str)
parser.add_argument("--exp", default="Test", type=str)
parser.add_argument("--tricks", default="None", type=str)
parser.add_argument("--batch-train", default=32, type=int)
parser.add_argument("--batch-val", default=32, type=int)
args = parser.parse_args()

def train(model, trn_loader, criterion, optimizer, epoch, mode="classification"):
    trn_loss = 0
    start_time = time.time()
    sum_total = 0 

    total_prob = np.zeros((0,20))
    total_target = np.zeros((0,20))
    for i, (image, target, file_name) in enumerate(trn_loader) :
        model.train()
        x = image.cuda()
        y = target.cuda()
        y_pred = model(x)  
        ious = np.zeros((21,))
        
        total_target = np.concatenate((total_target, target))

        if mode == "segmentation" : 
            loss = criterion(y_pred, y.long())
            pred= F.softmax(y_pred, dim= 1)
            _, max_index = torch.max(pred, 1)
            index_flatten = max_index.view(-1).cpu()
            target_flatten = target.view(-1).cpu()
            li = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            iou = jaccard_score(index_flatten, target_flatten, labels=li ,average=None)
            ious += iou
            measure = 0

        elif mode == "classification" :
            loss = criterion(y_pred, y)
            pos_probs = torch.sigmoid(y_pred)
            total_prob = np.concatenate((total_prob, pos_probs.detach().cpu().numpy()))
#             pos_preds = (pos_probs > 0.5).float()
#             tmp = torch.eq(pos_preds.cpu(), target.cpu()).sum().item()
#             tm1 = len(target) * 20
#             acc = tmp/tm1
#             sum_total += acc  
#             measure = acc

            #precision_score(pos_preds.cpu(), target.cpu(), average="macro") 
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                       
        trn_loss += (loss)

        end_time = time.time()
#         print(" [Training] [{0}] [{1}/{2}] Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}] Measure = [{5:.3f}]".format(epoch, i, len(trn_loader), loss.item(), end_time-start_time, measure))
        print(" [Training] [{0}] [{1}/{2}]".format(epoch, i, len(trn_loader)))
        start_time = time.time()

    trn_loss = trn_loss/len(trn_loader)
#     total_measure = sum_total / len(trn_loader)
    total_measure = average_precision_score(total_target, total_prob)

    if mode == "segmentation" : 
        total_measure = ious

    if epoch == 24 or epoch == 49 or epoch == 74 or epoch == 99  or epoch == 124 or epoch == 149 or epoch == 174 or epoch == 199: 
        torch.save(model.state_dict(), '{0}{1}_{2}_{3}.pth'.format("./", 'model', epoch, args.exp))

    print(" [Training] [{0}] [{1}/{2}] Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}] Measure = [{5:.3f}]".format(epoch, i+1, len(trn_loader), trn_loss, end_time - start_time, total_measure))
    return trn_loss, total_measure

def validate(model, val_loader, criterion, criterion_fn, optimizer, epoch, mode="segmentation"):
    val_loss = 0 
    model.eval()
    adv_losses = 0
    start_time = time.time()
    sum_total = 0 
    ious = np.zeros((21, ))

    total_prob = np.zeros((0, 20))
    total_target = np.zeros((0, 20))
    if args.method == 'adv' :
        for i, (data, target, file_name) in enumerate(val_loader) :
            x = data.cuda()
            y = target.cuda()
            
            total_target = np.concatenate((total_target, target))
            
            x_clone = Variable(x.clone().detach(), requires_grad=True).cuda()
            model_fn = copy.deepcopy(model)

            if mode == "segmentation" : 
                loss_fn = criterion_fn(model_fn(x_clone), y.long())

            elif mode == "classification" :
                loss_fn = criterion_fn(model_fn(x_clone), y)

            optimizer.zero_grad()
            loss_fn.backward()

            grads = x_clone.grad

            # calculate perturbations of the inputs
            scaled_perturbation = optimize_linear(grads, eps=0.25, norm=np.inf)
            # make adversarial samples
            adv_x = Variable(x_clone+scaled_perturbation, requires_grad=True).cuda()
            
            adv_y_pred = model_fn(adv_x)

            if mode == "segmentation" : 
                adv_loss = criterion_fn(adv_y_pred, y.detach().long())
                pred= F.softmax(adv_y_pred, dim= 1)
                _, max_index = torch.max(pred, 1)
                index_flatten = max_index.view(-1).cpu()
                target_flatten = target.view(-1).cpu()
                li = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                iou = jaccard_score(index_flatten, target_flatten, labels=li, average=None)
                ious += iou
                measure = 0 

            elif mode == "classification" :
                
                pos_probs = torch.sigmoid(adv_y_pred)
                pos_preds = (pos_probs > 0.5).float()
                adv_loss = criterion_fn(adv_y_pred, y.detach())
                total_prob = np.concatenate((total_prob, pos_probs.detach().cpu().numpy()))
                
#                 tmp = torch.eq(pos_preds.cpu(), target.cpu()).sum().item()
#                 tm1 = len(target) * 20
#                 acc = tmp/tm1
#                 sum_total += acc  
#                 measure = acc

            adv_losses += adv_loss.item()
            
            optimizer.zero_grad()

            end_time = time.time()
#             print(" [Validation] [{0}] [{1}/{2}] Adv. Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}] Measure [{5:.3f}]".format(epoch, i, len(val_loader), adv_loss.item(), end_time-start_time, measure))
            print(" [Validation] [{0}] [{1}/{2}]".format(epoch, i+1, len(val_loader)))
            start_time = time.time()
    else  :
        with torch.no_grad() :
            for i, (data, target, file_name) in enumerate(val_loader) :
                x = data.cuda()
                y = target.cuda()
                
                total_target = np.concatenate((total_target, target))
                
                y_pred = model(x)

                if mode == "segmentation" : 
                    loss = criterion(y_pred, y.long())
                    pred= F.softmax(y_pred, dim= 1)
                    _, max_index = torch.max(pred, 1)
                    index_flatten = max_index.view(-1).cpu()
                    target_flatten = target.view(-1).cpu()
                    li = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                    iou = jaccard_score(index_flatten, target_flatten, labels=li, average=None)
                    ious += iou
                    measure = 0 

                elif mode == "classification" :
                    loss = criterion(y_pred, y)
                    pos_probs = torch.sigmoid(y_pred)
                    pos_preds = (pos_probs > 0.5).float()
                    total_prob = np.concatenate((total_prob, pos_probs.detach().cpu().numpy()))
                    
#                     tmp = torch.eq(pos_preds.cpu(), target.cpu()).sum().item()
#                     tm1 = len(target) * 20
#                     acc = tmp/tm1
#                     sum_total += acc  
#                     measure = acc

                val_loss += (loss)

                end_time = time.time()
#                 print(" [Validation] [{0}] [{1}/{2}] Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}] Measure [{5:.3f}]".format(epoch, i, len(val_loader), loss.item(), end_time-start_time, measure))
                print(" [Validation] [{0}] [{1}/{2}]".format(epoch, i+1, len(val_loader)))
                start_time = time.time()


    if args.method != 'adv' :
        val_loss = val_loss / len(val_loader)
#         mean_total = sum_total / len(val_loader)
        mean_total = average_precision_score(total_target, total_prob)
        if mode == "segmentation" : 
            mean_total = ious 
        print(" [Validation] [{0}] [{1}/{2}] Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}] Measure [{5:.3f}]".format(epoch, i+1, len(val_loader), val_loss, end_time - start_time, mean_total))
        return val_loss, mean_total

    else : 
        adv_losses /= len(val_loader)
        mean_total = sum_total / len(val_loader)
        if mode == "segmentation" : 
            mean_total = ious 
        print(" [Validation] [{0}] [{1}/{2}] Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}] Measure [{5:.3f}]".format(epoch, i+1, len(val_loader), adv_losses, end_time - start_time, mean_total))
        return adv_losses, mean_total
    
def main():
    if args.mode == "segmentation" :
        label_path = "seg_da/VOCdevkit/VOC2010/SegmentationClass/"
        image_path = "seg_da/VOCdevkit/VOC2010/JPEGImages"

        if args.tricks == "cut-out" :
            trainset = dataset.voc_seg(label_path, image_path, cut_out=True, smooth=False)
            valset = dataset.voc_seg(label_path, image_path, cut_out=False, smooth=False)
        elif args.tricks == "smooth" :
            trainset = dataset.voc_seg(label_path, image_path, cut_out=False, smooth=True)
            valset = dataset.voc_seg(label_path, image_path, cut_out=False, smooth=False)
        elif args.tricks == "all" :
            trainset = dataset.voc_seg(label_path, image_path, cut_out=True, smooth=True)
            valset = dataset.voc_seg(label_path, image_path, cut_out=False, smooth=False)
        else :
            trainset = dataset.voc_seg(label_path, image_path, cut_out=False, smooth = False)
        
        total_idx = list(range(len(trainset)))
        split_idx = int(len(trainset) * 0.7)
        trn_idx = total_idx[:split_idx]
        val_idx = total_idx[split_idx:]

    elif args.mode == "classification" :
        info_path = "seg_da/VOCdevkit/VOC2010/SegmentationClass/"
        image_path = "seg_da/VOCdevkit/VOC2010/JPEGImages"

        if args.tricks == "smooth" :
            trainset = dataset.voc_cls(info_path, image_path, cut_out=False, smooth = True)
            valset = dataset.voc_cls(info_path, image_path, cut_out=False, smooth = False)
        elif args.tricks == "cut-out" :
            trainset = dataset.voc_cls(info_path, image_path, cut_out=True, smooth=True)
            valset = dataset.voc_cls(info_path, image_path, cut_out=False, smooth=False)
        elif args.tricks == "all" :
            trainset = dataset.voc_cls(info_path, image_path, cut_out=True, smooth =True)
            valset = dataset.voc_cls(info_path, image_path, cut_out=False, smooth=False)
        else :
            trainset = dataset.voc_cls(info_path, image_path, cut_out=False, smooth=False)

        total_idx = list(range(len(trainset)))
        split_idx = int(len(trainset) * 0.7)
        trn_idx = total_idx[:split_idx]
        val_idx = total_idx[split_idx:]
    else : 
        raise NotImplementedError

    if args.tricks == "smooth" or args.tricks == "cut-out" or args.tricks == "all" :
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_train, shuffle=False, sampler=SubsetRandomSampler(trn_idx))
        testloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_val, shuffle=False, sampler=SubsetRandomSampler(val_idx))
    else :
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_train, shuffle=False, sampler=SubsetRandomSampler(trn_idx))
        testloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_val, shuffle=False, sampler=SubsetRandomSampler(val_idx))

    if args.mode == "segmentation" :
        net = Unet2D((3, 256, 256), 1, 0.1, num_classes=21)
    elif args.mode == "classification" :
        net = torchvision.models.resnet50(pretrained=False, num_classes=20)
    else : 
        raise NotImplementedError

    if args.optim == 'sgd' :
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    elif args.optim == 'adam' :
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    elif args.optim == 'radam' :
        optimizer = RAdam(net.parameters(), lr = 0.0001)
    else : 
        raise NotImplementedError

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
    tr_acc = [] 
    val_losses = []
    val_acc = []
    for epoch in range(args.epochs) : 
        tr, tac = train(net, trainloader, criterion, optimizer, epoch, mode=args.mode)
        va, vac = validate(net, testloader, criterion, criterion_fn, optimizer, epoch, mode=args.mode)
        losses.append(tr)
        tr_acc.append(tac)
        val_losses.append(va)
        val_acc.append(vac)

    return losses, tr_acc, val_losses, val_acc


if __name__ == '__main__':
    tr_loss, tr_acc, val_loss, val_acc = main()
    try :
        torch.save(tr_loss, args.mode + args.method + str(args.epochs) + "Trainloss.pkl")
        torch.save(tr_acc, args.mode + args.method + str(args.epochs) + "Trainacc.pkl")
        torch.save(val_loss, args.mode + args.method + str(args.epochs) + "Validation.pkl")
        torch.save(val_acc, args.mode + args.method + str(args.epochs) + "Validation_acc.pkl")
    except : 
        import ipdb; ipdb.set_trace()
    
