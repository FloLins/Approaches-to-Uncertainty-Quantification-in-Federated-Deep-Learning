from networks import FeedForwardModel as net
import numpy as np
import torch
import utilities.config
import utilities.factory as factory


def run_main(MODEL_PARAMETERS, CONTROLL_PARAMETERS, seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    #Create Worker
    list_of_worker = factory.create_worker(MODEL_PARAMETERS, CONTROLL_PARAMETERS, net, seed)

    for i in range(MODEL_PARAMETERS['number_of_communications']):
        factory.compute_communication_less_period(list_of_worker)


if __name__ == "__main__":
    MODEL_PARAMETERS, CONTROLL_PARAMETERS = utilities.config.load_config()
    MODEL_PARAMETERS['lr'] = 0.1
    CONTROLL_PARAMETERS['optimisation'] = False
    for i in range(10):
        run_main(MODEL_PARAMETERS, CONTROLL_PARAMETERS, i)
