import training
import utilities.testing as testing
import utilities.config
import utilities.dataLoader as dataLoader
from networks import FeedForwardModel as net
from networks.EnsembleModel import ensembleModel
import torch
import numpy as np
import matplotlib.pyplot as plt


MODEL_PARAMETERS, CONTROLL_PARAMETERS = utilities.config.load_config()
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

    models = list()
    for j in range(MODEL_PARAMETERS['amount_of_communicators']):
        model = net.SimpleMNISTFeedForwardNet(MODEL_PARAMETERS)
        path = "models/" + str(seed) + "/model_from_Communicator-" + str(j) + "_" + str(
            MODEL_PARAMETERS['dataset']) + "_" + str(
            MODEL_PARAMETERS['epochs_per_episode']) + ".bin"
        model.load_state_dict(torch.load(path))
        model = model.to(device)
        models.append(model)

    ens_model = ensembleModel(models, MODEL_PARAMETERS)

    all_predictions, _, _, _ = ens_model.test_model(validation_set, MODEL_PARAMETERS['prediction_number'])
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
