import numpy as np
import torch


class Communicator:
    def __init__(self, name, model, len_worker, seed):
        print("Communicator Created")
        self.name = name
        self.model = model
        self.len_worker = len_worker
        self.received_models = []
        self.seed = seed
        #Swag Parameters
        self.swag_model = None
        self.counter = 0
        self.theta_swa = None
        self.swag_diag = list()

    def send_model(self):
        return self.model

    def receive_model(self, model):
        self.received_models.append(model)

    def clear_model(self):
        self.received_models = []

    def average_models(self):
        states = []
        for model in self.received_models:
            states.append(model.state_dict())
        if len(self.received_models) > 0:
            average = states[0]
            states.pop(0)
            for state in states:
                weights = [0] * len(state.items())
                counter = 0
                for key, value in state.items():
                    weights[counter] = state[key]
                    counter = counter +1
                counter = 0
                for key, value in average.items():
                    average[key] = np.add(average[key].cpu(), weights[counter].cpu())
                    counter = counter + 1
            for key, value in average.items():
                average[key] = np.true_divide(average[key], len(self.received_models))
            self.model.load_state_dict(average)

    def save_model(self):
        torch.save(self.model.state_dict(),  f'''models/{self.seed}/model_from_{self.name}_{self.model.MODEL_PARAMETER['dataset']}_{self.model.MODEL_PARAMETER['epochs_per_episode']}.bin''')

    def take_one(self):
        self.model = self.received_models[0]

    #Swag Operations
    def receive_swag(self, counter, new_theta_swa, new_swag_diag):
        self.counter = self.counter + counter
        if self.theta_swa is None:
            self.theta_swa = new_theta_swa
        else:
            self.theta_swa = np.add(self.theta_swa, new_theta_swa)
        self.swag_diag.append(new_swag_diag)

    def set_swag_values_to_model(self):
        self.swag_model.counter = self.counter
        self.swag_model.theta_swa = self.theta_swa
        self.swag_model.swag_diag = self.swag_diag
        print("Set Values")

    def compute_swag_values(self):
        length = self.len_worker
        self.theta_swa = np.true_divide(self.theta_swa, length)

        new_swag_diag = self.swag_diag[0]
        for i in range(1, length):
            new_swag_diag = np.add(new_swag_diag, self.swag_diag[i])
        div = 1 / np.square(length)
        new_swag_diag = np.multiply(new_swag_diag, div)
        self.swag_diag = new_swag_diag

    def save_swag_model(self):
        self.swag_model.save_model()
