
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
parser.add_argument('--data_dir', metavar='DIR', help='path to load model')
parser.add_argument('--model_name', type=str, help='name of the model to load')
parser.add_argument('--time_file', type=str, help='name of the model to load')
parser.add_argument('--model', default=0, type=int, metavar='N',
                    help='0=squeezenet, 1=vgg16_bn')
parser.add_argument('--workers', default=2, type=int, metavar='N')
parser.add_argument('--print_frequency', default=100, type=int, metavar='N')
parser.add_argument('--epochs', default=1, type=int, metavar='N')
parser.add_argument('--batch_size', default=100, type=int, metavar='N')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR')
parser.add_argument('--decay_coef', default=30, type=int, metavar='N')
parser.add_argument('--noten', dest='ten', action='store_false')

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

def replace_relu(model):
    for i in range(len(model.features)):
        if isinstance(model.features[i], nn.ReLU):
            model.features[i] = nn.Softshrink(lambd=0.5)

def model_bucket_avg(model):
    for i in range(len(model.features)):
        if isinstance(model.features[i], Conv.ALSHConv2d):
            print(model.features[i].avg_bucket_freq())

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
    elif args.model == 2:
        vgg_cifar(version=models.vgg11)
    else:
        vgg_cifar(version=models.alexnet)

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
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True,
        transform=train_transform)

    val_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True,
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.workers,
        pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False,
	    num_workers=args.workers, pin_memory=False)

    criterion = nn.CrossEntropyLoss()

    time_file_path = '/data/zhanglab/afeeney/times/'
    time_file = open(time_file_path + args.time_file, 'w+')
    avg_time_file = open(time_file_path + args.time_file + '_avg', 'w+')

    # train while replacing every replace_gap
    depth = 7 # vgg11=8, alexnet = 4

    model = model.cpu()
    #replace_relu(model) # goes through the model replacing relu with whatever.
    current_depth = len(model.features)-1
    flag = True
    for _ in range(depth):
        current_depth = replace_next_conv(model, current_depth)
        model.features[current_depth + 1].first = True
        if flag == True:
            model.features[current_depth + 1].last = True
            flag = False
        #model.features[current_depth + 1].last = True
        model.apply(fix)
        avg_batch_time = validate(val_loader, model, criterion, time_file)
        avg_time_file.write(str(avg_batch_time) + '\n')
        model.features[current_depth+1].first = False

    model.features[current_depth+1].first = True

    #model_bucket_avg(model)

    end = time.time()
    avg_batch_time = validate(val_loader, model, criterion, time_file)
    val_time = time.time() - end
    avg_time_file.write(str(avg_batch_time) + '\n')

    print('\n\ncifar10:' if args.ten else 'cifar100:')
    print('ALSH Conv Time Test; no retraining')
    print(' * Original Model: ' + args.model_name)
    print('\n')
    print(' * CPU Val Time:       ' + str(val_time))
    print(' * Average Batch Time: ' + str(avg_batch_time))



def validate(val_loader, model, criterion, time_file):
    batch_time = AverageMeter()

    model.eval()

    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cpu()
            target = target.cpu()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            end = time.time()
            output = model(input_var)
            batch_time.update(time.time() - end)

            if i > 0:
                time_file.write(str(batch_time.val) + '\n')

        if i == 21:
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
