import numpy as np
import torch.nn.functional as F
from networks.BaseModel import Model
import torch
from utilities import functions as fx


class ensembleModel(Model):
    def __init__(self, models, MODEL_PARAMETER):
        super(ensembleModel, self).__init__(MODEL_PARAMETER)
        self.models = models

    def to(self, device):
        models = []
        for model in self.models:
            test = model.to(device)
            models.append(test)
        self.models = models

    def predict(self, X_test, pred_number=1):
        preds = []
        for mod in self.models:
            pred = mod.forward(X_test)
            pred = F.softmax(pred, dim=1).cpu().data.numpy()
            preds.append(pred)
        return np.array(preds)





