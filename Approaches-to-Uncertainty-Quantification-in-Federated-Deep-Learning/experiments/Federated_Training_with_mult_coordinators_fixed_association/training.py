from networks import FeedForwardModel as net
import numpy as np
import torch
from utilities import factory
import utilities.config


def run_main(MODEL_PARAMETERS, CONTROLL_PARAMETERS, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    #Create Worker
    list_of_worker = factory.create_worker(MODEL_PARAMETERS, CONTROLL_PARAMETERS, net, seed)

    #Create Communicator
    list_of_communicator = factory.create_communicator(MODEL_PARAMETERS, CONTROLL_PARAMETERS, net, seed)

    worker_per_communicator = int(MODEL_PARAMETERS['amount_of_worker']/MODEL_PARAMETERS['amount_of_communicators'])
    orderd_lists_of_worker = []
    #print(list_of_worker)
    for i in range(MODEL_PARAMETERS['amount_of_communicators']):
        subgroup_of_worker = list_of_worker[i*worker_per_communicator:(i+1)*worker_per_communicator]
        orderd_lists_of_worker.append(subgroup_of_worker)
    #print(orderd_lists_of_worker)

    for i in range(MODEL_PARAMETERS['number_of_communications']):
        for j in range(MODEL_PARAMETERS['amount_of_communicators']):
            factory.compute_one_communication_period(orderd_lists_of_worker[j], list_of_communicator[j])


if __name__ == "__main__":
    MODEL_PARAMETERS, CONTROLL_PARAMETERS = utilities.config.load_config()
    MODEL_PARAMETERS['lr'] = 0.1
    CONTROLL_PARAMETERS['optimisation'] = False
    for i in range(10):
        run_main(MODEL_PARAMETERS, CONTROLL_PARAMETERS, i)