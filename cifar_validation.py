
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
parser.add_argument('--model', default=0, type=int, metavar='N', help='0=squeezenet, 1=vgg16_bn')
parser.add_argument('--workers', default=2, type=int, metavar='N')
parser.add_argument('--print_frequency', default=100, type=int, metavar='N')
parser.add_argument('--epochs', default=1, type=int, metavar='N')
parser.add_argument('--batch_size', default=100, type=int, metavar='N')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, metavar='LR')
parser.add_argument('--decay_coef', default=30, type=int, metavar='N')
parser.add_argument('--ten', default=True, type=bool,
                    help='if True, it will use CIFAR10, if False, CIFAR100')


def main():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    global args, best_prec1, best_prec5

    args = parser.parse_args()

    if args.model == 0:
        model = models.squeezenet1_1(pretrained=True)
        model.num_classes=10 if args.ten else 100
        model.classifier[1] = nn.Conv2d(512, model.num_classes, kernel_size=1, stride=1)
    elif args.model == 1:
        model = models.vgg16_bn(pretrained=True)
        model.num_classes=10 if args.ten else 100
        model.classifier[-1] = nn.Linear(4096, model.num_classes)

    train_sampler = None

    normalize = transforms.Normalize(mean=(0.485,0.456,0.406),
                                     std=(0.229,0.224,0.225))

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    # valset is really the test set. Wanted to keep name consistent with ImageNet version.
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

    model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    ftrs_to_opt = [#{'params': model.features[-1].parameters()},
                   #{'params': model.features[-2].parameters()},
                   {'params': model.classifier.parameters()}]


    #optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9)
    optimizer = optim.SGD(ftrs_to_opt, args.lr, momentum=0.9)
    #optimizer = optim.Adam(ftrs_to_opt, args.lr)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    end = time.time()

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)

        # this is a "validation" test. It has no effect on when the real test happens.
        prec1, prec5 = validate(val_loader, model, criterion)
        best_prec1 = max(best_prec1, prec1.item())
        best_prec5 = max(best_prec5, prec5.item())

    train_time = time.time() - end
    end = time.time()

    # this is the "real" test. The time it occurs is irrelavent
    top1, top5 = validate(val_loader, model, criterion)

    val_time = time.time() - end

    print('\n')
    print('cifar10:' if args.ten else 'cifar100:')
    print('\n')
    print('This test used: \t {epochs} Epochs, \t {bs} Batch Size, \t {lr} Initial LR, \t  {dc} Decay Coef'.\
          format(epochs=args.epochs, bs=args.batch_size, lr=args.lr, dc=args.decay_coef))
    print(' * Total Train Time:   ' + str(train_time))
    print(' * Final top1 Val Acc: ' + str(top1.item()))
    print(' * Final top5 Val Acc: ' + str(top5.item()))
    print(' * Best top1 Val Acc:  ' + str(best_prec1))
    print(' * best top5 Val Acc:  ' + str(best_prec5))
    print(' * CPU Val Time:       ' + str(val_time))


    file_name = 'cifar{num}_{model}_{acc:.2f}'.\
        format(num=(10 if args.ten else 100), model=('squeezenet' if args.model == 0 else 'vgg16'),
               acc=top1.item())
    path = '/data/zhanglab/afeeney/models/' + file_name
    torch.save(model, path)

    print('model saved to ' + path)



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_frequency == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.val:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, top1=top1, top5=top5))


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
            input = input.cuda()
            target = target.cuda()

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

    return top1.avg, top5.avg


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

if __name__ == "__main__":
    main()
