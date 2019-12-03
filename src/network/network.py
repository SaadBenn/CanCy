import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np


batch_size = 32
epoch = 100


class TorchDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)

        assert self.inputs.shape[0] == self.targets.shape[0]

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]

        return X, y


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, random_state=42):
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        super(Network, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.layer1 = None  # nn.Linear(input_shape, 64), we don't know what's the input size yet
        self.layer2 = nn.Linear(64, 16)

        # output
        self.layer3 = nn.Linear(16, 16)
        self.output_layer = None  # nn.Linear(16, output_shape), we don't know what's the output size yet

        # reconstruction (autoencoder)
        self.pre_reconstruct_layer = nn.Linear(16, 64)
        self.reconstruct_output = nn.Linear(64, input_shape)

        self.output_criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.parameters(), lr=0.001)

    def __initialize_network(self, input_shape, output_shape):
        self.layer1 = nn.Linear(input_shape, 64)
        self.output_layer = nn.Linear(16, output_shape)

    def forward(self, inputs):
        layer1 = F.relu(self.layer1(inputs))
        layer2 = F.relu(self.layer2(layer1))

        # output
        layer3 = F.relu(self.layer3(layer2))
        output = F.sigmoid(self.output_layer(layer3))

        # reconstruction
        pre_reconstruct_layer = F.relu(self.pre_reconstruct_layer(layer2))
        reconstruct_output = self.reconstruct_output(pre_reconstruct_layer)

        return output, reconstruct_output

    def fit(self, inputs, targets):
        self.__initialize_network(inputs.shape[0], targets)
        self.train(True)
        # make dataset to torch type
        dataset = TorchDataset(inputs, targets)
        data_loader = DataLoader(dataset, batch_size, shuffle=False)  # shuffle false because data already shuffled

        for batch_x, batch_y in data_loader:
            variable_batch_x = Variable(batch_x)
            variable_batch_y = Variable(batch_y)

            output, reconstruction = self.forward(variable_batch_x)

    def score(self, inputs, targets):
        pass
