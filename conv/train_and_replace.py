import sys
sys.path.append('../')
sys.path.append('../conv/')

import time
from utils import AverageMeter, accuracy
import torch
import torch.nn as nn


def train_and_replace(train_loader,
                      model,
                      criterion,
                      optimizer,
                      depth=None,
                      gap=None,
                      device=torch.device('cuda')):
    '''
    uses train and replace strategy to take an existing well-trained CNN
    and make it an ALSH CNN
    '''
    flag = True

    # train and replace the top layer.
    # Doing it separately simplifies the logic for deeper layers.
    #model.features[current_depth + 1].last = True
    #for epoch in range(gap):
    #    train(train_loader, model, criterion, optimizer, epoch, device)

    # train and replace the deeper layers.
    for epoch in range((depth) * gap):
        if epoch % gap == 0:
            # replacing a new layer, so the current 'first' ALSHConv2d
            # will now be the second. So, set first to False.
            #if current_depth < len(model.features) - 1:
            #    model.features[current_depth + 1].first = False
            model.features[current_depth + 1].use_naive()
            current_depth = replace_next_conv(model, current_depth)
        model.apply(fix)

        train(train_loader, model, criterion, optimizer, epoch, device)
    model.features[current_depth + 1].first = True

    return model, optimizer


def replace_next_conv(model, current):
    '''
    from the current index, this function coes down to the next convolution.
    It finds it and replaces it.
    returns the index of the conv - 1
    '''

    # find the next convolution
    while not isinstance(model.features[current], nn.Conv2d):
        current -= 1

    if isinstance(model.features[current], nn.Conv2d):
        print('replaced with ALSHConv2d at Feature: ' + str(current))
        model.features[current] = Conv.ALSHConv2d.build(
            model.features[current], MultiHash_SRP, {}, 5, 3, 2**5)

    # current-1 so the next call to this function will not start
    # at a convolution.
    return current - 1


def train(train_loader, model, criterion, optimizer, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # training defaults to GPU
        input = input.to(device)
        target = target.to(device)

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
            print('Train-And-Replace: [{0}][{1}/{2}]\t'
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
