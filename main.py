
import torch
import torchvision # to get cifar10 data for testing
import torchvision.transforms as transforms

import torch.optim as optim
import alsh_net as ALSH
import alsh_conv_net as ALSHConv

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5,.5,.5), (.5,.5,.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)

    net = ALSHConv.ALSHConvNet().cuda()

    train(net, trainloader, 1)

    correct, total = test(net, testloader)

    print( (correct / total) * 100)



def train(net, trainloader, num_epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, net.parameters()),
                    lr=.0001)

    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            inputs.resize_(32, 3, 32, 32)

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
