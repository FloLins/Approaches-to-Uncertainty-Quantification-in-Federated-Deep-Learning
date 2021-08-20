import utilities.config
from networks import FeedForwardModel as net
from networks.SwagTrainModel import EncapsulatedModel
from networks.SwagTestModel import SWAGNet
import numpy as np
import torch
from utilities import dataLoader
import utilities.config
from utilities import plots, testing, printing

np.random.seed(2020)
torch.manual_seed(2020)


def run_main():
    np.random.seed(2020)
    torch.manual_seed(2020)
    MODEL_PARAMETERS, CONTROLL_PARAMETERS = utilities.config.load_config()

    CONTROLL_PARAMETERS['Train'] = False
    test_set = dataLoader.load_test_set(dataset=MODEL_PARAMETERS['dataset'])
    test_set_ood = dataLoader.load_test_set('KMNIST')

    models_all_seeds = []
    for i in range(10):
        model = net.SimpleMNISTFeedForwardNet(MODEL_PARAMETERS)
        name = "Communicator-0"
        encapsulated_model = EncapsulatedModel(model, name, i)
        encapsulated_model.load_model()
        print("Counter " + str(encapsulated_model.counter))

        mu = encapsulated_model.theta_swa
        mu = encapsulated_model.restore_orig_shape(mu)
        var = encapsulated_model.swag_diag
        var = encapsulated_model.restore_orig_shape(var)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        swag_net = SWAGNet(mu, var, MODEL_PARAMETERS)
        swag_net.to(device)
        models_all_seeds.append(swag_net)

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
        plots.plot_entropy_and_variance(in_data["entropy"], right_wrong["entropy_right"], right_wrong["entropy_wrong"], ood_data["entropy"], in_data["variance"], ood_data["variance"], "Federated Training with SWAG")
        plots.plot_calibration_curve(calibration['curve_label'], calibration['softmax_pred'], "Federated Training with SWAG")


if __name__ == "__main__":
    run_main()

