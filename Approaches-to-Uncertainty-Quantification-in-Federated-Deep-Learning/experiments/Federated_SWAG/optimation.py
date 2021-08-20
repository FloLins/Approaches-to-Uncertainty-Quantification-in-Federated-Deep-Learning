import training
import utilities.config
import utilities.dataLoader as dataLoader
from networks import FeedForwardModel as net
from networks.SwagTestModel import SWAGNet
from networks.SwagTrainModel import EncapsulatedModel
import torch
import numpy as np
import matplotlib.pyplot as plt


MODEL_PARAMETERS, CONTROLL_PARAMETERS = utilities.config.load_config()
CONTROLL_PARAMETERS['swag'] = True
CONTROLL_PARAMETERS['optimisation'] = True
all_trainsets  = utilities.dataLoader.load_data_and_datasets(MODEL_PARAMETERS['amount_of_worker']+1, dataset=MODEL_PARAMETERS['dataset'])
validation_set = all_trainsets[-1]
all_accuracy = []
all_rates = []
seed = 2021

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lrs = [0.001, 0.005, 0.01, 0.05, 0.1]
for lr in lrs:
    all_rates.append(lr)
    MODEL_PARAMETERS['lr'] = lr
    training.run_main(MODEL_PARAMETERS, CONTROLL_PARAMETERS, seed)

    model = net.SimpleMNISTFeedForwardNet(MODEL_PARAMETERS)
    name = "Communicator-0"
    encapsulated_model = EncapsulatedModel(model, name, seed)
    encapsulated_model.load_model()
    print("Counter " + str(encapsulated_model.counter))

    mu = encapsulated_model.theta_swa
    mu = encapsulated_model.restore_orig_shape(mu)
    var = encapsulated_model.swag_diag
    var = encapsulated_model.restore_orig_shape(var)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    swag_net = SWAGNet(mu, var, MODEL_PARAMETERS)
    swag_net.to(device)

    all_predictions, _, _, _ = swag_net.test_model(validation_set, MODEL_PARAMETERS['prediction_number'])
    accuracy = all_predictions['accuracy']
    all_accuracy.append(accuracy)

max_accuracy_index = np.argmax(all_accuracy)
rate = all_rates[int(max_accuracy_index)]
accuracy = all_accuracy[int(max_accuracy_index)]
print("Best Results with learning rate: " + str(rate) + " with a accuracy of: " + str(accuracy))
print(max_accuracy_index)

plt.plot(all_accuracy)
plt.ylabel('accuracy')
plt.show()