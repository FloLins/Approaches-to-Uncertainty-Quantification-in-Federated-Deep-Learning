from networks import DropOutFeedForwardModel as net
import numpy as np
import torch
from utilities import dataLoader
import utilities.config
from utilities import plots, testing
from utilities import printing


np.random.seed(2020)
torch.manual_seed(2020)


def run_main():
    np.random.seed(2020)
    torch.manual_seed(2020)
    MODEL_PARAMETERS, CONTROLL_PARAMETERS = utilities.config.load_config()

    CONTROLL_PARAMETERS['Train'] = False
    CONTROLL_PARAMETERS['dropOut'] = True
    test_set = dataLoader.load_test_set(dataset=MODEL_PARAMETERS['dataset'])
    test_set_ood = dataLoader.load_test_set('KMNIST')

    models_all_seeds = []
    for i in range(10):
        path = "models/" + str(i) + "/model_from_Communicator-0_MNIST_10.bin"
        model = net.SimpleMNISTDropOutFeedForwardNet(MODEL_PARAMETERS)
        model.load_state_dict(torch.load(path))
        models_all_seeds.append(model)

    in_data, ood_data, right_wrong, aurocs, calibration = testing.test_multiple_seeds(models_all_seeds, test_set, test_set_ood,
                                                                                MODEL_PARAMETERS['prediction_number'])

    #Print Results
    print("########################################")
    print("RESULTS IN-DATA")
    printing.print_data(in_data)
    print("RESULTS OOD-DATA")
    printing.print_data(ood_data)
    print("RIGHT-WRONG_RESULTS")
    printing.print_right_wrong(right_wrong)
    print("AUROCS")
    printing.print_aurocs(aurocs)

    if CONTROLL_PARAMETERS['plotting']:
        plots.plot_entropy_and_variance(in_data["entropy"], right_wrong["entropy_right"], right_wrong["entropy_wrong"], ood_data["entropy"], in_data["variance"], ood_data["variance"], "Federated Training with MC-Dropout")
        plots.plot_calibration_curve(calibration['curve_label'], calibration['softmax_pred'], "Federated Training with MC-Dropout")


if __name__ == "__main__":
    run_main()
