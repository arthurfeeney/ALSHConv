import sys
sys.path.append('/data/zhanglab/afeeney/ALSHNN/')

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

import conv.alsh_conv_2d as Conv
from lsh.multi_hash_srp import MultiHash_SRP

import time

best_prec1 = 0
best_prec5 = 0

parser = argparse.ArgumentParser(description='arguments for CIFAR validation')
parser.add_argument('--data_dir', metavar='DIR', help='path to the dataset')
parser.add_argument('--workers', default=2, type=int, metavar='N')
parser.add_argument('--print_frequency', default=100, type=int, metavar='N')
parser.add_argument('--epochs', default=1, type=int, metavar='N')
parser.add_argument('--batch_size', default=100, type=int, metavar='N')
parser.add_argument('--lr',
                    '--learning_rate',
                    default=0.1,
                    type=float,
                    metavar='LR')
parser.add_argument('--decay_coef', default=30, type=int, metavar='N')
parser.add_argument('--replace_gap', default=10, type=int, metavar='N')
parser.add_argument('--depth', default=4, type=int, metavar='N')
parser.add_argument('--final_epochs', default=20, type=int, metavar='N')


def replace_next_conv(model, current):
    while not isinstance(model.features[current], nn.Conv2d):
        current -= 1
    if isinstance(model.features[current], nn.Conv2d):
        print('REPLACED REGULAR Conv2d WITH ALSHConv2d')
        model.features[current] = Conv.ALSHConv2d.build(
            model.features[current], MultiHash_SRP, {}, 5, 3, 2**5)
    return current - 1


def fix(m):
    if isinstance(m, Conv.ALSHConv2d):
        m.fix()
        m.cuda()


def to_cpu(m):
    if isinstance(m, Conv.ALSHConv2d):
        m = m.cpu()


def model_reset_stats(model):
    for i in range(len(model.features)):
        if isinstance(model.features[i], Conv.ALSHConv2d):
            model.features[i].reset_freq()


def model_bucket_avg(model):
    for i in range(len(model.features)):
        if isinstance(model.features[i], Conv.ALSHConv2d):
            avg = str(model.features[i].avg_bucket_freq())
            sum = str(model.features[i].sum_bucket_freq())
            print('avg of hashes: ' + avg + '; sum of hashes: ' + sum)


def init_weight(l):
    if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
        nn.init.kaiming_normal_(l.weight)


class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.Hardshrink(lambd=0.3),
            #nn.Softshrink(),
            #nn.ReLU(),
            #nn.Tanh(),
            #nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.Hardshrink(lambd=0.3),
            #nn.Softshrink(),
            #nn.ReLU(),
            #nn.Tanh(),
            #nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(3872, num_classes),
                                        nn.Softmax(dim=1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def main():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    global args, best_prec1, best_prec5

    args = parser.parse_args()

    train_sampler = None

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))

    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(28),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        #transforms.Resize(32),
        #transforms.CenterCrop(28),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = torchvision.datasets.MNIST(root=args.data_dir,
                                               train=True,
                                               download=True,
                                               transform=train_transform)

    val_dataset = torchvision.datasets.MNIST(root=args.data_dir,
                                             train=False,
                                             download=True,
                                             transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=False)

    model_name = 'modelsMNIST_custommodel_98.47'
    model = Model()  #torch.load('/data/zhanglab/afeeney/models/' + model_name)
    print(model)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), args.lr)

    current_depth = len(model.features) - 1
    flag = True
    for epoch in range(args.depth * args.replace_gap):
        if epoch % args.replace_gap == 0:
            if current_depth < len(model.features) - 1:
                model.features[current_depth + 1].first = False
            current_depth = replace_next_conv(model, current_depth)
            if flag:
                model.features[current_depth + 1].last = True
                flag = False
            model.features[current_depth + 1].first = True
        model.apply(fix)

        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)

    # the last replacement is the first ALSHConv2d in the network!
    model.features[current_depth + 1].first = True

    for epoch in range(args.final_epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)

    model_reset_stats(model)
    top1, top5, avg_batch_time = validate(val_loader, model, criterion)
    model_bucket_avg(model)

    print('\n MNIST Accuracy: \n')
    print(' * Top1: ' + str(top1))
    print(' * Top5: ' + str(top5))
    print('\n')

    #file_name = 'MNIST_custommodel_{acc:.2f}'.format(acc=top1.item())
    #path = '/data/zhanglab/afeeney/models/' + file_name
    #torch.save(model, path)
    #print('model saved to ' + path)


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

        input = input.cuda()
        target = target.cuda()

        input = input.reshape(input.size(0), 1, 28, 28)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

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
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5))


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
            input = input.reshape(input.size(0), 1, 28, 28)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

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
                          i,
                          len(val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1,
                          top5=top5))

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


def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1**(epoch // args.decay_coef))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_ALSH_mode(m):
    if isinstance(m, Conv.ALSHConv2d):
        m.ALSH_mode()


def sqz_cifar():
    model = models.squeezenet1_1(pretrained=True)
    model.num_classes = 10 if args.ten else 100
    model.classifier[1] = nn.Conv2d(512,
                                    model.num_classes,
                                    kernel_size=1,
                                    stride=1)
    return model


def vgg_cifar(version):
    model = version(pretrained=True)
    model.num_classes = 10 if args.ten else 100
    model.classifier[-1] = nn.Linear(4096, model.num_classes)
    return model


if __name__ == "__main__":
    main()
