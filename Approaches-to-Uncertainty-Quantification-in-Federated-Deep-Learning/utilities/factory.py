import utilities.dataLoader
from utilities.worker import Worker
from utilities.communicator import Communicator
from networks.SwagTrainModel import EncapsulatedModel


def create_worker(MODEL_PARAMETERS, CONTROLL_PARAMETERS, net, seed):
    if CONTROLL_PARAMETERS['optimisation'] is False:
        datasets = utilities.dataLoader.load_data_and_datasets(MODEL_PARAMETERS['amount_of_worker'],
                                                           dataset=MODEL_PARAMETERS['dataset'])
    else:
        datasets = utilities.dataLoader.load_data_and_datasets(MODEL_PARAMETERS['amount_of_worker']+1, dataset=MODEL_PARAMETERS['dataset'])
    list_of_worker = []
    for i in range(MODEL_PARAMETERS['amount_of_worker']):
        name = "Worker-" + str(i)
        data = datasets[i]
        network = net.create_new_network(MODEL_PARAMETERS)
        if CONTROLL_PARAMETERS['diffrerent_dropout_rates']:
            droprate = MODEL_PARAMETERS['dropout_rate'] + i * MODEL_PARAMETERS['dropout_rate_increase']
            droprate = droprate % 1
            network.drop_rate = droprate
        new_worker = Worker(name, network, data, MODEL_PARAMETERS, CONTROLL_PARAMETERS, seed)
        list_of_worker.append(new_worker)
    return list_of_worker


def create_communicator(MODEL_PARAMETERS, CONTROLL_PARAMETERS, net, seed):
    list_of_communicator = []
    for i in range(MODEL_PARAMETERS['amount_of_communicators']):
        name = "Communicator-" + str(i)
        network = net.create_new_network(MODEL_PARAMETERS)
        amount_worker = MODEL_PARAMETERS['amount_of_worker']
        communicator = Communicator(name, network, amount_worker, seed)
        if CONTROLL_PARAMETERS['swag']:
            communicator.swag_model = EncapsulatedModel(network, name, seed)
        list_of_communicator.append(communicator)
    return list_of_communicator


def compute_one_communication_period(list_of_worker, communicator):
    communicator.len_worker = len(list_of_worker) #important for random association
    for worker in list_of_worker:
        worker.model.set_loader(worker.data)
        worker.train_model()
        new_model = worker.send_model()
        communicator.receive_model(new_model)
    communicator.average_models()
    communicator.save_model()
    new_model = communicator.send_model()
    for worker in list_of_worker:
        worker.receive_model(new_model)
    print("[+] FINISHED PERIOD")


def compute_communication_less_period(list_of_worker):
    for worker in list_of_worker:
        worker.model.set_loader(worker.data)
        worker.train_model()


def compute_swag_period(list_of_worker, communicator):
    for worker in list_of_worker:
        worker.model.set_loader(worker.data)
        worker.encapsulate_swag_model()
        worker.swag_train_model()
        counter, new_theta_swa, new_swag_diag = worker.send_swag()
        communicator.receive_swag(counter, new_theta_swa, new_swag_diag)
    communicator.compute_swag_values()
    communicator.set_swag_values_to_model()
    communicator.save_swag_model()
