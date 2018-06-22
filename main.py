
import torch
import torchvision # to get cifar10 data for testing
import torchvision.transforms as transforms

import torch.optim as optim
import Net.alsh_net as ALSH
import Net.alsh_conv_net as ALSHConv
import Net.alsh_alex_net as ALSHAlex
import Net.alsh_vgg_net as ALSHVGG
import Net.simp_vgg_net as SimpVGG

import time

def main():
    transform = transforms.Compose(
        [#transforms.Resize((227,227)),
         transforms.ToTensor(),
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

    net = ALSHConv.ALSHConvNet().cuda()
    #net = ALSHVGG.ALSHVGGNet().cuda()
    #net = SimpVGG.SimpVGGNet().cuda()

    start = time.time()

    train(net, trainloader, 1)

    train_time = time.time() - start

    correct, total = test(net, testloader)

    test_time = time.time() - start - train_time

    total_time = time.time() - start

    print( (correct / total) * 100)

    print('times: ')
    print('train: ', train_time)
    print('test: ', test_time)
    print('total: ', total_time)



def train(net, trainloader, num_epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, net.parameters()),
                    lr=.001)

    net.train()

    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            inputs.resize_(100, 3, 32, 32)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

        print('epoch: ', epoch)

    print('training complete')


def test(net, testloader):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            inputs.resize_(1, 3, 32, 32)

            outputs = net(inputs, False)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct, total


if __name__ == "__main__":
    main()
