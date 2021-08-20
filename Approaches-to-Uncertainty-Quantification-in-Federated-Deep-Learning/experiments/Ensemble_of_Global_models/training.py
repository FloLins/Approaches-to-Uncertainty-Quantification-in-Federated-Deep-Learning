from networks import FeedForwardModel as net
import numpy as np
import torch
import utilities.config
import utilities.factory as factory


def run_main(MODEL_PARAMETERS, CONTROLL_PARAMETERS, seed):
    for i in range(0, MODEL_PARAMETERS['repetitions_for_ensemble']):
        np.random.seed(seed+i*10)
        torch.manual_seed(seed+i*10)
        name = "Communicator-" + str(i)
        #Create Worker
        list_of_worker = factory.create_worker(MODEL_PARAMETERS, CONTROLL_PARAMETERS, net, seed)


        #Create Communicator
        list_of_communicator = factory.create_communicator(MODEL_PARAMETERS, CONTROLL_PARAMETERS, net, seed)
        communicator = list_of_communicator[0]
        communicator.name = name

        for i in range(MODEL_PARAMETERS['number_of_communications']):
            factory.compute_one_communication_period(list_of_worker, communicator)


if __name__ == "__main__":
    MODEL_PARAMETERS, CONTROLL_PARAMETERS = utilities.config.load_config()
    MODEL_PARAMETERS['lr'] = 0.1
    CONTROLL_PARAMETERS['optimisation'] = False
    for i in range(10):
        run_main(MODEL_PARAMETERS, CONTROLL_PARAMETERS, i)
