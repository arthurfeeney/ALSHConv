
import torch
import torchvision # to get cifar10 data for testing
import torchvision.transforms as transforms

import torch.optim as optim
import Net.alsh_net as ALSH
import Net.alsh_conv_net as ALSHConv
import Net.alsh_alex_net as ALSHAlex
import Net.alsh_vgg_net as ALSHVGG
import Net.normal_vgg_net as NormVGG

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

    device = torch.device('cpu')

    #net = ALSHConv.ALSHConvNet(device).to(device)

    net = ALSHVGG.ALSHVGGNet(device).to(device)
    #net = NormVGG.NormVGGNet().to(device)

    start = time.time()

    train(net, trainloader, 1, device)

    train_time = time.time() - start

    #correct, total = test(net, testloader, device)

    #test_time = time.time() - start - train_time

    #total_time = time.time() - start

    #print( (correct / total) * 100)

    #print('times: ')
    #print('train: ', train_time)
    #print('test: ', test_time)
    #print('total: ', total_time)



def train(net, trainloader, num_epochs, device=torch.device('cuda')):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, net.parameters()),
                    lr=.001)

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

            if i == 10:
                break
        break
        print('epoch: ', epoch)



    end = time.time()

    time_span = end - start

    print('training complete: ', time_span)


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


if __name__ == "__main__":
    main()
