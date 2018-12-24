import torch
import torch.nn as nn
import torchvision # to get cifar10 data for testing
import torchvision.transforms as transforms
import torchvision.models as models

import torch.optim as optim
import Conv.alsh_conv_2d as Conv
from LSH.multi_hash_srp import MultiHash_SRP

import time


def init_net(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')

def set_ALSH_mode(m):
    if isinstance(m, Conv.ALSHConv2d):
        m.ALSH_mode()

def setup_SqueezeNet():

    #
    # Replace existing Conv2d with ALSHConv2d and copy the weights
    #
    return net


def main():
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.485,0.456,0.406), 
                              std=(0.229,0.224,0.225))])

    trainset = torchvision.datasets.CIFAR10(
                                root='./Data/cifar-10',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=20,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
                                root='./Data/cifar-10',
                                           train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                             shuffle=True, num_workers=2)

    device = torch.device('cpu')

    #net = models.vgg16_bn(pretrained=True)
    #net.classifier[-1] = nn.Linear(4096, 10)
    
    
    net = models.squeezenet1_1(pretrained=True)
    net.num_classes=10
    net.classifier[1] = nn.Conv2d(512, 10, kernel_size=1, stride=1)

    ftr_weight = net.features[-1].expand3x3.weight 
    repl = Conv.ALSHConv2d(64, 256, 3, 1, 1, 1, True, MultiHash_SRP, {}, 
                           3, 2, 2**3) 
    repl.weight = ftr_weight
    net.features[-1].expand3x3 = repl
    net.apply(set_ALSH_mode)





    #
    # Transfer Learning with CIFAR10
    #
    start = time.time()
    
    #net.classifier.train()
    #train(net, trainloader, 1, device)
    
    train_time = time.time() - start
    print('train time: ', train_time)


    #
    # Save the model
    # 
    #torch.save(net.state_dict(), 'CIFAR10_Models/vgg16_bn') 


    #
    # Evaluate model on test dataset
        #
    #net.apply(set_ALSH_mode)

    net.eval()
    correct, total = test(net, testloader, device)
    test_time = time.time() - start - train_time
    
    total_time = time.time() - start
    print( (correct / total) * 100)
    print('times: ')
    print('train: ', train_time)
    print('test: ', test_time)
    print('total: ', total_time)

# END main


def train(net, trainloader, num_epochs, device=torch.device('cuda')):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=.1, momentum=.9,
                                weight_decay=.0001)

    net.train()

    start = time.time()

    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs.resize_(20, 3, 224, 224)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            if i == 400:
                return 1
                print('epoch: ' + str(epoch) + 'finished iteration: ' + str(i))

 
        adjust_learning_rate(.1, optimizer, epoch)

        print('epoch: ', epoch)


def test(net, testloader, device=torch.device('cuda')):
    correct = 0
    total = 0

    net.eval()

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs.resize_(20, 3, 224, 224)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct, total

def adjust_learning_rate(learning_rate, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    with batch size of 100, this is dividing learning_rate by 10 every
    15k iterations.
    """
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
