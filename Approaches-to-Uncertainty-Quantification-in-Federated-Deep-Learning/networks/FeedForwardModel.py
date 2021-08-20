import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.BaseModel import Model


class SimpleMNISTFeedForwardNet(Model):
    def __init__(self, MODEL_PARAMETER):
        super(SimpleMNISTFeedForwardNet, self).__init__(MODEL_PARAMETER)
        input_dim= 784
        output_dim = 10
        h1_dim=128
        h2_dim=128
        self.fc_xh1 = nn.Linear(input_dim, h1_dim)
        self.fc_h1h2 = nn.Linear(h1_dim, h2_dim)
        self.fc_h2y = nn.Linear(h2_dim, output_dim)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.reshape(x, [x.shape[0], 784]).to(device) #the last bit of data in an epoch may be smaller than batchSize, so x.shape[0] is most of the time==batchsize
        h1 = F.relu(self.fc_xh1(x))
        h2 = F.relu(self.fc_h1h2(h1))
        y = self.fc_h2y(h2)
        return y


class SimpleCifar10FeedForwardNet(Model):
    def __init__(self, MODEL_PARAMETER):
        super(SimpleCifar10FeedForwardNet, self).__init__(MODEL_PARAMETER)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_new_network(MODEL_PARAMETERS):
    if MODEL_PARAMETERS['dataset'] == "MNIST":
        return SimpleMNISTFeedForwardNet(MODEL_PARAMETERS)
    elif MODEL_PARAMETERS['dataset'] == "CIFAR10":
        return SimpleCifar10FeedForwardNet(MODEL_PARAMETERS)