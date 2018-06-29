
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesVGG9Net(nn.Module):
    def __init__(self):
        super(BayesVGG9Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 1, 1, 1, bias=False)
        self.drop1 = nn.Dropout2d()
        self.conv2 = nn.Conv2d(64, 64, 1, 1, 1, bias=False)
        self.drop2 = nn.Dropout2d()
        self.pool1 = nn.MaxPool2d(2)


        self.conv3 = nn.Conv2d(64, 128, 1, 1, 1, bias=False)
        self.drop3 = nn.Dropout2d()
        self.conv4 = nn.Conv2d(128, 128, 1, 1, 1, bias=False)
        self.drop4 = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(2)


        self.conv5 = nn.Conv2d(128, 256, 1, 1, 1, bias=False)
        self.drop5 = nn.Dropout2d()
        self.conv6 = nn.Conv2d(256, 256, 1, 1, 1, bias=False)
        self.drop6 = nn.Dropout2d()
        self.pool3 = nn.MaxPool2d(2)

        self.fc7 = nn.Linear(4096, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = self.drop1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.drop2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.drop3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.drop4(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.drop5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.drop6(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.view(batch_size, -1)

        x = self.fc7(x)
        x = F.relu(x)

        x = self.fc8(x)
        x = F.relu(x)

        x = self.fc9(x)

        return x
