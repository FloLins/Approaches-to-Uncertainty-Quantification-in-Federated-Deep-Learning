import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utilities import functions as fx


class Model(nn.Module):
    def __init__(self, MODEL_PARAMETER):
        super(Model, self).__init__()
        self.MODEL_PARAMETER = MODEL_PARAMETER
        self.trainloader = None

    def set_loader(self, trainset):
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.MODEL_PARAMETER['batch_size'],
                                                       shuffle=True, num_workers=0)

    def get_params(self):
        return self.MODEL_PARAMETER

    def train_model(self):
        #Cuda
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self = self.to(device)

        #Params
        MODEL_PARAMETER = self.get_params()
        lr = MODEL_PARAMETER['lr']  # learning rate
        lam = MODEL_PARAMETER['lam']  # weight decay
        epochs = MODEL_PARAMETER['epochs_per_episode'] #epochs per communication episode

        #Optimizer and Loss
        opt = optim.SGD(self.parameters(), lr=lr, weight_decay=lam)
        CrEnt = nn.CrossEntropyLoss()

        for epoch in range(epochs):  # loop over the dataset multiple times
            for _, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                opt.zero_grad()
                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = CrEnt(outputs, labels)
                loss.backward()
                opt.step()
            print(loss.item())

    def predict(self, X_test, pred_number=1):
        if X_test.shape[0]>50000:
            print('implement batch mode')
        else:
            preds = []
            for _ in range(pred_number):
                pred = self.forward(X_test)
                pred = F.softmax(pred, dim=1).cpu().data.numpy()
                preds.append(pred)
        return np.array(preds)

    def test_model(self, test_set, pred_number):
        testloader = torch.utils.data.DataLoader(test_set, batch_size=self.MODEL_PARAMETER['batch_size'],
                                                 shuffle=False, num_workers=0)
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        all_outputs = None
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.predict(images, pred_number)

                #Stack all Outputs for further Usage
                if all_outputs is None:
                    all_outputs = torch.Tensor(outputs)
                else:
                    all_outputs = torch.cat([all_outputs, torch.Tensor(outputs)], dim=1)

                # the class with the highest energy is what we choose as prediction
                predicted = np.argmax(np.mean(outputs, axis=0), axis=1)
                predicted = torch.Tensor(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                #list all labels and predictions
                all_predictions = all_predictions + predicted.tolist()
                all_labels = all_labels + labels.tolist()

        #Accuracy Emtropy and Variance
        accuracy = correct / total
        entropy = fx.calculate_entropy(all_outputs)
        variance = np.max(fx.calculate_variance(all_outputs), axis=1)
        #Softmax of Output
        softmax_pred = np.max(np.mean(all_outputs.numpy(), axis=0), axis=1)


        #Split Right and Wrong Predictions
        pred_right = []
        pred_wrong = []
        curve_label = []

        for i in range(len(all_predictions)):
            if all_predictions[i] - all_labels[i] == 0:
                to_append = all_outputs[:, i, :].numpy()
                pred_right.append(to_append)
                curve_label.append(1)
            else:
                to_append = all_outputs[:, i, :].numpy()
                pred_wrong.append(to_append)
                curve_label.append(0)

        pred_right = np.array(pred_right)
        pred_wrong = np.array(pred_wrong)
        pred_right = np.swapaxes(pred_right, 0, 1)
        pred_wrong = np.swapaxes(pred_wrong, 0, 1)
        #Entropy and Variance for Right and Wrong Predictions
        entropy_right = fx.calculate_entropy(pred_right)
        entropy_wrong = fx.calculate_entropy(pred_wrong)
        variance_right = np.max(fx.calculate_variance(pred_right), axis=1)
        variance_wrong = np.max(fx.calculate_variance(pred_wrong), axis=1)

        all_predictions = {
            "accuracy": accuracy,
            "entropy": entropy,
            "variance": variance
        }
        right_predictions = {
            "entropy": entropy_right,
            "variance": variance_right
        }
        wrong_predictions = {
            "entropy": entropy_wrong,
            "variance": variance_wrong
        }
        calibration = {
            "curve_label": curve_label,
            "softmax_pred": softmax_pred
        }
        return all_predictions, right_predictions, wrong_predictions, calibration