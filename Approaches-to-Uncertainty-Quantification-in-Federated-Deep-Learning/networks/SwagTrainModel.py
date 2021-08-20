import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class EncapsulatedModel:
    def __init__(self, model, ID, seed, debug=False):
        print("[+] Encapsulate Model in Swag Model")
        self.ID = ID
        self.seed = seed
        self.model = model
        self.counter = 1
        weights = np.array(list(model.parameters()))

        self.orig_shape = self.save_orig_shape(weights)
        if debug:
            print("Orig_Shape " + str(self.orig_shape))
            print("Orig_Type" + str(type(weights)))
        weights = self.local_get_weights_array(weights)
        self.new_shape = len(weights)
        if debug:
            print("New Shape of Array: " + str(self.new_shape))

        self.theta_swa = weights
        if debug:
            print("Init Shape Theta swa: " + str(self.theta_swa.shape))

        self.theta_line_squared = self.square_weights(weights)
        if debug:
            print("Init Shape Theta Line squared: " + str(self.theta_line_squared))

        self.swag_diag = None

    def get_params(self):
        return self.model.get_params()

    def save_model(self, debug=False):
            if debug:
                print("Try to save model parameters")
            torch.save(self.theta_swa, 'models/' + str(self.seed) + '/swag/theta_swa' + self.ID + '.pt')
            torch.save(self.theta_line_squared, 'models/'+ str(self.seed) + '/swag/theta_line_squared' + self.ID + '.pt')
            torch.save(self.swag_diag, 'models/' + str(self.seed) + '/swag/swag_diag' + self.ID)
            torch.save(self.model.state_dict(), 'models/'+ str(self.seed) + '/swag/swag_model' + self.ID + '.bin')
            torch.save(self.counter, 'models/' + str(self.seed) + '/swag/counter' + self.ID + '.pt')
            torch.save(self.orig_shape, 'models/' + str(self.seed) + '/swag/orig_shape' + self.ID + '.pt')
            torch.save(self.new_shape, 'models/' + str(self.seed) + '/swag/new_shape' + self.ID + '.pt')

    def load_model(self, debug=True):
            if debug:
                print("Try to load model parameters")
            self.theta_swa = torch.load('models/' + str(self.seed) + '/swag/theta_swa' + self.ID + '.pt')
            self.theta_line_squared = torch.load('models/' + str(self.seed) + '/swag/theta_line_squared' + self.ID + '.pt')
            self.swag_diag = torch.load('models/' + str(self.seed) + '/swag/swag_diag' + self.ID)
            self.model.load_state_dict(torch.load('models/' + str(self.seed) + '/swag/swag_model' + self.ID + '.bin'))
            self.counter = torch.load('models/' + str(self.seed) + '/swag/counter' + self.ID + '.pt')
            self.orig_shape = torch.load('models/' + str(self.seed) + '/swag/orig_shape' + self.ID + '.pt')
            self.new_shape = torch.load('models/' + str(self.seed) + '/swag/new_shape' + self.ID + '.pt')

    def save_orig_shape(self, weights, debug=False):
            if debug:
                print("Type of Weights Array is: " + str(type(weights)))
                print("Shape of Weights Array is: " + str(weights.shape))
            shapes = []
            for entry in weights:
                if debug:
                    print("Type of Entry" + str(type(entry)))
                shapes.append(entry.shape)
            return shapes

    def restore_orig_shape(self, weights, debug=False):
            output = []
            for entry in self.orig_shape:
                if debug:
                    print("For entry : " + str(entry))
                    print("Size of weight Array: " + str(len(weights)))
                if len(entry) == 2:
                    i = entry[0]
                    j = entry[1]
                    product = i*j
                    if debug:
                        print("Will take first X elements out of weights. X=" + str(product))
                    used_weights = weights[:product]
                    weights = weights[product:]
                    result = np.reshape(used_weights, (i, j))
                    result = torch.tensor(np.array(result))
                    if debug:
                        print("Resulting Shape " + str(result.shape))
                    result = torch.nn.Parameter(result)
                    output.append(result)
                if len(entry) == 1:
                    i = entry[0]
                    if debug:
                        print("Will take first X elements out of weights. X=" + str(i))
                    used_weights = weights[:i]
                    weights = weights[i:]
                    result = torch.tensor(np.array(used_weights))
                    if debug:
                        print("Resulting Shape " + str(result.shape))
                    result = torch.nn.Parameter(result)
                    output.append(result)
            output = np.asarray(output)
            if debug:
                print("Type of Output: " + str(type(output)))
                print("Shape of Output: " + str(output.shape))
                print("Type of Each Element: " + str(type(output[0])))
            return output

    def forward(self, x):
        y = self.model.forward(x)
        return y

    def divide(self):
        self.theta_swa = np.true_divide(self.theta_swa, self.counter)
        self.theta_line_squared = np.true_divide(self.theta_line_squared, self.counter)

    def compute_swag_diagonal(self, debug=False):
        self.divide()
        theta_swa_squared = self.square_weights(self.theta_swa)
        sub = np.subtract(self.theta_line_squared, theta_swa_squared)
        super_threshold_indices = sub < 0
        sub[super_threshold_indices] = 0
        self.swag_diag = sub

        if debug:
            print("Shape Theta swa: " + str(self.theta_swa.shape))
            print("Shape Theta swa squared: " + str(theta_swa_squared.shape))
            print("Shape Swag_Diag: " + str(sub.shape))
            min_element = np.amin(self.swag_diag)
            max_element = np.amax(self.swag_diag)
            print(str(min_element)+ " at " + str(np.argmin(self.swag_diag)))
            print(str(max_element)+ " at " +str(np.argmax(self.swag_diag)))

    def add_to_swa(self, new_weights):
        self.theta_swa = np.add(self.theta_swa, new_weights)

    def add_to_line(self, new_weights):
        new_weights = self.square_weights(new_weights)
        self.theta_line_squared = np.add(self.theta_line_squared, new_weights)

    def square_weights(self, weights):
        return np.multiply(weights, weights)

    def get_weights_array(self, weights):
        array = weights[0].data
        array = array.flatten()
        try:
            array = torch.from_numpy(array)
        except:
            x = 0
        array = array.cpu()
        for dim in range(1, len(weights)):
            tensor = weights[dim].data
            tensor = tensor.flatten().cpu()
            array = np.concatenate((array, tensor), 0)
        return array

    def local_get_weights_array(self, weights):
        weights = weights
        array = weights[0].data
        array = array.flatten()
        for dim in range(1, len(weights)):
            tensor = weights[dim].data
            tensor = tensor.flatten()
            array = np.concatenate((array, tensor), 0)
        return array

    def train_model(self):
        #Cuda
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        #Params
        MODEL_PARAMETER = self.get_params()
        lr = MODEL_PARAMETER['lr']  # learning rate
        lam = MODEL_PARAMETER['lam']  # weight decay
        epochs = MODEL_PARAMETER['epochs_per_episode'] #epochs per communication episode

        #Optimizer and Loss
        opt = optim.SGD(self.model.parameters(), lr=lr, weight_decay=lam)
        CrEnt = nn.CrossEntropyLoss()

        for epoch in range(epochs):  # loop over the dataset multiple times
            for _, data in enumerate(self.model.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                opt.zero_grad()
                # forward + backward + optimize
                outputs = self.model.forward(inputs)
                loss = CrEnt(outputs, labels)
                loss.backward()
                opt.step()
                # Here Swag Magic
                self.counter = self.counter + 1
                new_weights = np.array(list(self.model.parameters()))
                new_weights = self.get_weights_array(new_weights)
                self.add_to_swa(new_weights)
                self.add_to_line(new_weights)
            print(loss.item())
        self.compute_swag_diagonal()
        return self
