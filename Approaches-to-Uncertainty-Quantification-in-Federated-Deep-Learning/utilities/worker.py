import torch
from networks.SwagTrainModel import EncapsulatedModel


class Worker:
    def __init__(self, name, model, data, MODEL_PARAMETERS, CONTROLL_PARAMETERS, seed, swag_model=None):
        print("Worker Created")
        self.name = name
        self.model = model
        self.swag_model = swag_model
        self.MODEL_PARAMETERS = MODEL_PARAMETERS
        self.CONTROLL_PARAMETERS = CONTROLL_PARAMETERS
        self.data = data
        self.seed = seed

    def send_model(self):
        if self.CONTROLL_PARAMETERS['dropOut']:
            self.model = self.model.to("cpu")
            self.model.rescale()
        self.model = self.model.to("cpu")
        return self.model

    def receive_model(self, model):
        if self.CONTROLL_PARAMETERS['dropOut']:
            self.model = model.to("cpu")
            self.model.backscale()
        else:
            self.model = model

    def train_model(self):
        '''
        returns a trained torch model
        '''
        if self.CONTROLL_PARAMETERS['Train']:
            print('[+] TRAINING MODEL..\n')
            self.model.train_model()
            torch.save(self.model.state_dict(), f'''models/{self.seed}/model_from_{self.name}_{self.MODEL_PARAMETERS['dataset']}_{self.MODEL_PARAMETERS['epochs_per_episode']}.bin''')
        else:
            print('[+] LOAD MODEL..')
            self.model.load_state_dict(torch.load(
                        f'''models/{self.seed}/model_from_{self.name}_{self.MODEL_PARAMETERS['dataset']}_{self.MODEL_PARAMETERS['epochs_per_episode']}.bin'''))

    #Swag Functions
    def send_swag(self):
        counter = self.swag_model.counter
        theta_swa = self.swag_model.theta_swa
        swag_diag = self.swag_model.swag_diag
        return counter, theta_swa, swag_diag

    def swag_train_model(self):
        # returns a swag-trained torch model
        if self.CONTROLL_PARAMETERS['Train']:
            print('[+] SWAG TRAINING MODEL..\n')
            self.swag_model = self.swag_model.train_model()
            self.swag_model.save_model()
        else:
            print('[+] LOAD MODEL..')
            self.swag_model.load_model()

    def encapsulate_swag_model(self):
        self.model = self.model.cpu()
        self.swag_model = EncapsulatedModel(self.model, self.name, self.seed)
