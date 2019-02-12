
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import gc
from PIL import ImageFile

import Conv.alsh_conv_2d as Conv
from LSH.multi_hash_srp import MultiHash_SRP

import time

best_prec1 = 0
best_prec5 = 0

parser = argparse.ArgumentParser(description='arguments for CIFAR validation')
parser.add_argument('model_path', metavar='DIR', help='path to load model')
parser.add_argument('--model_name', type=str, help='name of the model to load')
parser.add_argument('--model', default=0, type=int, metavar='N',
                    help='0=squeezenet, 1=vgg16_bn')
parser.add_argument('--workers', default=2, type=int, metavar='N')
parser.add_argument('--print_frequency', default=100, type=int, metavar='N')
parser.add_argument('--epochs', default=1, type=int, metavar='N')
parser.add_argument('--batch_size', default=100, type=int, metavar='N')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR')
parser.add_argument('--decay_coef', default=30, type=int, metavar='N')
parser.add_argument('--ten', default=True, type=bool,
                    help='if True, it will use CIFAR10, if False, CIFAR100')

def replace_conv(model, idx):
    if isinstance(model.features[idx], models.squeezenet.Fire):
        model.features[idx].squeeze =  Conv.ALSHConv2d.build(
            model.features[idx].squeeze, MultiHash_SRP, {}, 5, 4, 2**5)
        model.features[idx].expand1x1 = Conv.ALSHConv2d.build(
            model.features[idx].expand1x1, MultiHash_SRP, {}, 5, 4, 2**5)
        model.features[idx].expand3x3 = Conv.ALSHConv2d.build(
            model.features[idx].expand3x3, MultiHash_SRP, {}, 5, 4, 2**5)

def replace_next_conv(model, current):
    while not isinstance(model.features[current], nn.Conv2d):
        current -= 1
    if isinstance(model.features[current], nn.Conv2d):
        print('REPLACED REGULAR Conv2d WITH ALSHConv2d')
        model.features[current] = Conv.ALSHConv2d.build(
            model.features[current], MultiHash_SRP, {}, 5, 3, 2**5)
    return current-1

def fix(m):
    if isinstance(m, Conv.ALSHConv2d):
        m.fix()
        m = m.cpu()

def main():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    global args, best_prec1, best_prec5

    args = parser.parse_args()

    if args.model == 0:
        model = sqz_cifar()
    elif args.model == 1:
        vgg_cifar(version=models.vgg16_bn)
    else:
        vgg_cifar(version=models.vgg11)

    model = torch.load(args.model_path + args.model_name,
                       map_location=lambda storage, loc: storage )
    model = model.module.cpu()

    model.apply(fix)

    train_sampler = None

    normalize = transforms.Normalize(mean=(0.485,0.456,0.406),
                                     std=(0.229,0.224,0.225))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    # valset is really the test set.
    if args.ten:
        train_dataset = torchvision.datasets.CIFAR10(
            root='./Data/cifar-10', train=True, download=True,
            transform=train_transform)

        val_dataset = torchvision.datasets.CIFAR10(
            root='./Data/cifar-10', train=False, download=True,
            transform=val_transform)
    else:
        train_dataset = torchvision.datasets.CIFAR100(
            root='./Data/cifar-100', train=True, download=True,
            transform=train_transform)

        val_dataset = torchvision.datasets.CIFAR100(
            root='./Data/cifar-100', train=False, download=True,
            transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.workers,
        pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False,
	    num_workers=args.workers, pin_memory=False)

    criterion = nn.CrossEntropyLoss()

    # train while replacing every replace_gap
    depth = 8 # vgg11=9, vgg16 features=18, sqznet uses 10

    model = model.cpu()
    current_depth = 0
    for _ in range(depth):
        current_depth = replace_next_conv(model, current_depth)

    model.apply(fix)


    end = time.time()
    top1, top5, avg_batch_time = validate(val_loader, model, criterion)
    val_time = time.time() - end

    print('\n\ncifar10:' if args.ten else 'cifar100:')
    print('ALSH Conv Time Test; no retraining')
    print(' * Original Model: ' + args.model_name)
    print('\n')
    print(' * CPU Val Time:       ' + str(val_time))
    print(' * Average Batch Time: ' + str(avg_batch_time))



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    start = time.time()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cpu()
            target = target.cpu()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target, topk=(1,5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_frequency == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))
        if i == 10:
            return top1.avg, top5.avg, batch_time.avg


    return top1.avg, top5.avg, batch_time.avg


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk=max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // args.decay_coef))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_ALSH_mode(m):
    if isinstance(m, Conv.ALSHConv2d):
        m.ALSH_mode()

def sqz_cifar():
    model = models.squeezenet1_1(pretrained=True)
    model.num_classes=10 if args.ten else 100
    model.classifier[1] = nn.Conv2d(512, model.num_classes, kernel_size=1,
                                    stride=1)
    return model

def vgg_cifar(version):
    model = version(pretrained=True)
    model.num_classes=10 if args.ten else 100
    model.classifier[-1] = nn.Linear(4096, model.num_classes)
    return model

if __name__ == "__main__":
    main()