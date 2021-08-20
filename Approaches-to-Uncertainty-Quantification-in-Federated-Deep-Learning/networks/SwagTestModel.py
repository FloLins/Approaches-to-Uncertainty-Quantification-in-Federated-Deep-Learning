import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.BaseModel import Model


class BNNLinear(nn.Module):
    ''' Layer for a Bayesian Neural Net'''

    def __init__(self, in_dim, out_dim, mu, var):
        super(BNNLinear, self).__init__()

        in_dim += 1
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.mu = nn.Parameter(mu)
        self.var = nn.Parameter(var)

    def forward(self, x, shape, debug=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.reshape(x, [x.shape[0], shape]).to(device)  # the last bit of data in an epoch may be smaller than batchSize, so x.shape[0] is most of the time==batchsize
        m = x.size(0)
        r, c = self.in_dim, self.out_dim
        M = self.mu
        var = self.var

        E = torch.randn(m, *var.shape).to(device)
        if debug:
            print("m: " + str(m))
            print("Var Shape: " + str(var.shape))
        if var.shape[0]*var.shape[1]> r*c:
            if debug:
                print('non diagonal covariance')
            sigma = torch.bmm(torch.sqrt(var), torch.transpose(E, 2,3) )
        else:
            if debug:
                print('diagonal covariance')
            sigma = torch.sqrt(var)*E

        W = M + sigma
        W = W.transpose(1, 2)

        if debug:
            print("Sigma Shape: " + str(sigma.shape))
            print("W SHape: " + str(W.shape))
            print("x shape: " + str(x.shape))
        x = torch.cat([x.cpu(), torch.ones(m, 1)], 1)
        if debug:
            print("x shape: " + str(x.shape))
            print("x shape unsquessez1 : " + str(x.unsqueeze(1).shape))
        h = torch.bmm(x.unsqueeze(1), W.cpu()).squeeze(1)

        return h


class SWAGNet(Model):

    def __init__(self, mu, var, MODEL_PARAMETER):
        super(SWAGNet, self).__init__(MODEL_PARAMETER)
        input_dim = 784
        output_dim = 10
        h1_dim = 128
        h2_dim = 128
        var1 = torch.cat((var[0], var[1].unsqueeze(1)), 1)
        mu1 = torch.cat((mu[0], mu[1].unsqueeze(1)), 1)
        var2 = torch.cat((var[2], var[3].unsqueeze(1)), 1)
        mu2 = torch.cat((mu[2], mu[3].unsqueeze(1)), 1)
        var3 = torch.cat((var[4], var[5].unsqueeze(1)), 1)
        mu3 = torch.cat((mu[4], mu[5].unsqueeze(1)), 1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc_xh1 = BNNLinear(input_dim, h1_dim, mu1, var1).to(device)
        self.fc_h1h2 = BNNLinear(h1_dim, h2_dim, mu2, var2).to(device)
        self.fc_h2y = BNNLinear(h2_dim, output_dim, mu3, var3).to(device)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        h1 = self.fc_xh1(x, 784)
        h1 = F.relu(h1)
        h2 = self.fc_h1h2(h1, 128)
        h2 = F.relu(h2)
        y = self.fc_h2y(h2, 128)
        return y

    def predict(self, X_test, pred_number=1):
        '''
        INPUT: model and data to predict
        OUTPUT: a numpy array of size : pred_number, number of samples, number of output nodes
        '''

        if X_test.shape[0] > 50000:
            print('implement batch mode')
        else:
            preds = []
            for _ in range(pred_number):
                pred = self.forward(X_test)
                pred = F.softmax(pred, dim=1).cpu().data.numpy()
                preds.append(pred)

        return np.array(preds)