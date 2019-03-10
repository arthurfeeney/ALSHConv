
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
parser.add_argument('--time_file', type=str, help='name of the model to load')
parser.add_argument('--model', default=0, type=int, metavar='N',
                    help='0=squeezenet, 1=vgg16_bn')
parser.add_argument('--workers', default=2, type=int, metavar='N')
parser.add_argument('--noten', dest='ten', action='store_false')

def main():
    torch.set_num_threads(16)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    global args, best_prec1, best_prec5

    args = parser.parse_args()

    if args.model == 0:
        model = sqz_cifar()
    elif args.model == 1:
        model = vgg_cifar(version=models.vgg16_bn)
    elif args.model == 2:
        model = vgg_cifar(version=models.vgg11)
    else:
        vgg_cifar(version=models.alexnet)


    model = torch.load(args.model_path + args.model_name,
                       map_location=lambda storage, loc: storage )
    model = model.module.cpu()

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

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False,
	    num_workers=args.workers, pin_memory=False)

    criterion = nn.CrossEntropyLoss()
    model = model.cpu()

    time_file_path = '/data/zhanglab/afeeney/times/'
    time_file = open(time_file_path + args.time_file + '_NO_ALSH', 'w+')

    iters = 21 * 5

    end = time.time()
    avg_batch_time = validate(val_loader, model, criterion, time_file, iters)
    val_time = time.time() - end

    print('Base Time Test For CPU Validation')
    print('\n\ncifar10:' if args.ten else 'cifar100:')
    print(' * Original Model: ' + args.model_name)
    print('\n')
    print(' * CPU Val Time:       ' + str(val_time))
    print(' * Average Batch Time: ' + str(avg_batch_time))



def validate(val_loader, model, criterion, time_file, iters):
    batch_time = AverageMeter()

    model.eval()

    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            end = time.time()
            output = model(input_var)
            batch_time.update(time.time() - end)

            if i > 0:
                time_file.write(str(batch_time.val) + '\n')

        if i == iters:
            return batch_time.avg

    return batch_time.avg


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
