import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.BaseModel import Model


class SimpleMNISTDropOutFeedForwardNet(Model):
    def __init__(self, MODEL_PARAMETER):
        super(SimpleMNISTDropOutFeedForwardNet, self).__init__(MODEL_PARAMETER)
        input_dim= 784
        output_dim = 10
        h1_dim=128
        h2_dim=128
        self.fc_xh1 = nn.Linear(input_dim, h1_dim)
        self.fc_h1h2 = nn.Linear(h1_dim, h2_dim)
        self.fc_h2y = nn.Linear(h2_dim, output_dim)
        self.drop_rate = MODEL_PARAMETER['dropout_rate']

    def forward2(self, x):
        h1 = F.relu(self.fc_xh1(x))
        h1 = F.dropout(h1, p=self.drop_rate, training=True)
        h2 = F.relu(self.fc_h1h2(h1))
        h2 = F.dropout(h2, p=self.drop_rate, training=True)
        y = self.fc_h2y(h2)
        return y

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.reshape(x, [x.shape[0], 784]).to(
            device)  # the last bit of data in an epoch may be smaller than batchSize, so x.shape[0] is most of the time==batchsize
        h1 = self.fc_xh1(x)
        h1 = F.relu(h1)
        m = torch.distributions.Bernoulli(torch.full(h1.shape, 1-self.drop_rate))
        bern = m.sample().to(device)
        h1 = (h1 * bern)
        h2 = self.fc_h1h2(h1)
        h2 = F.relu(h2)
        m = torch.distributions.Bernoulli(torch.full(h2.shape, 1-self.drop_rate))
        bern = m.sample().to(device)
        h2 = h2 * bern
        y = self.fc_h2y(h2)
        return y

    def cpu_forward(self, x):
        h1 = F.relu(self.fc_xh1(x))
        m = torch.distributions.Bernoulli(torch.full(h1.shape, 1-self.drop_rate))
        bern = m.sample()
        h1 = h1 * bern
        h2 = F.relu(self.fc_h1h2(h1))
        m = torch.distributions.Bernoulli(torch.full(h2.shape, 1-self.drop_rate))
        bern = m.sample()
        h2 = h2 * bern
        y = self.fc_h2y(h2)
        return y

    def rescale(self):
        state = self.state_dict()
        for key, value in state.items():
            state[key] = np.multiply(state[key], (1-self.drop_rate))
        self.load_state_dict(state)

    def backscale(self):
        state = self.state_dict()
        for key, value in state.items():
            state[key] = np.true_divide(state[key], (1-self.drop_rate))
        self.load_state_dict(state)


def create_new_network(MODEL_PARAMETERS):
    return SimpleMNISTDropOutFeedForwardNet(MODEL_PARAMETERS)
