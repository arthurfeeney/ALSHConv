
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
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

parser = argparse.ArgumentParser(description='arguments for imagenet validation')
parser.add_argument('data', metavar='DIR', help='Path to dataset.')
parser.add_argument('--workers', default=2, type=int, metavar='N')
parser.add_argument('--print_frequency', default=100, type=int, metavar='N')
parser.add_argument('--epochs', default=1, type=int, metavar='N')
parser.add_argument('--devices', default=1, type=int, metavar='N')
parser.add_argument('--batch_size', default=128, type=int, metavar='N')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, metavar='LR')

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

    model = models.alexnet(pretrained=True)#models.squeezenet1_1(pretrained=True)

    ftr_weight = model.features[-1].expand3x3.weight
    r = Conv.ALSHConv2d(64, 256, 3, 1, 1, 1, True, MultiHash_SRP, {}, 5, 8, 2**5)
    r.weight = ftr_weight
    model.features[-1].expand3x3 = r

    ftr_weight = model.features[-2].expand3x3.weight
    r = Conv.ALSHConv2d(64, 256, 3, 1, 1, 1, True, MultiHash_SRP, {}, 5, 8, 2**5)
    r.weight = ftr_weight
    model.features[-2].expand3x3 = r

    model.apply(set_ALSH_mode)




    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')
    train_sampler = None

    normalize = transforms.Normalize(mean=[0.485, 0.465, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.workers,
        pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir,
                             transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize])),
	    batch_size=64, shuffle=False,
	    num_workers=args.workers, pin_memory=False)

    model = model.cuda()


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9)

    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)



    end = time.time()

    depth = 4
    current = 0

    for d in range(depth):
        replace_next_conv(model.features[current


    #for epoch in range(args.epochs):
    #    adjust_learning_rate(optimizer, epoch)
    #    train(train_loader, model, criterion, optimizer, epoch)

        #prec1, prec5 = validate(val_loader, model, criterion)
        #best_prec1 = max(prec1, best_prec1)
        #best_prec5 = max(prec1, best_prec5)

    train_time = time.time() - end
    end = time.time()

    validate(val_loader, model, criterion)

    val_time = time.time() - end

    print(' * Total Train Time:  ' + str(train_time))
    print(' * Best top1 Val Acc: ' + str(best_prec1))
    print(' * best top5 Val Acc: ' + str(best_prec5))
    print(' * CPU Val Time:      ' + str(val_time))
    print('\n')
    print('This test used: \t {epochs} Epochs, \t {bs} Batch Size, \t {lr} initial Learning Rate'.format(
          epochs=args.epochs, bs = args.batch_size, lr = args.learning_rate))



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

    # re-organize weights in ALSH table at the end of each epoch? Much faster than doing it every iteration.

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
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_ALSH_mode(m):
    if isinstance(m, Conv.ALSHConv2d):
        m.ALSH_mode()

if __name__ == "__main__":
    main()
