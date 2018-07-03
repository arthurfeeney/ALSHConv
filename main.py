
import torch
import torchvision # to get cifar10 data for testing
import torchvision.transforms as transforms

import torch.optim as optim
import Net.alsh_net as ALSH
import Net.alsh_conv_net as ALSHConv
import Net.alsh_alex_net as ALSHAlex
import Net.alsh_vgg_net as ALSHVGG
import Net.normal_vgg_net as NormVGG
import SingleCPUTests.standard_conv as std

import time


def init_net(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5,.5,.5), (.5,.5,.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)

    device = torch.device('cpu')

    net = ALSHConv.ALSHConvNet(device).to(device)
    #net = std.ManyFilterNet().to(device)
    #net = ALSHVGG.ALSHVGGNet(device).to(device)

    #net = NormVGG.NormVGGNet().to(device)
    #net.apply(init_net)

    start = time.time()

    train(net, trainloader, 200, device)

    train_time = time.time() - start

    correct, total = test(net, testloader, device)
    #correct, total = monte_carlo_dropout_test(net, testloader, device=device)

    test_time = time.time() - start - train_time

    total_time = time.time() - start

    print( (correct / total) * 100)

    print('times: ')
    print('train: ', train_time)
    print('test: ', test_time)
    print('total: ', total_time)



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

            inputs.resize_(100, 3, 32, 32)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

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

            inputs.resize_(1, 3, 32, 32)

            outputs = net(inputs, False)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct, total

def monte_carlo_test_phase(m):
    if type(m) != torch.nn.Dropout2d:
        m.eval()

def monte_carlo_dropout_test(net, testloader, T, device=torch.device('cuda')):

    r"""
    used with Bayesian networks. Averages T outputs for each input.
    """

    correct = 0
    total = 0

    net.apply(monte_carlo_test_phase) # switch to test mode, except dropout!


    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs.resize_(1, 3, 32, 32)

            outputs = torch.Tensor([net(inputs) for _ in range(T)]).to(device)

            final_output = outputs.sum(dim=0) / outputs.size()[0]

            _, predicted = torch.max(final_output.data, 1)
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
