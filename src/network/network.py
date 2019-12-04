import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np


batch_size = 32
epoch = 1000


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


class OnlyXDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = torch.from_numpy(inputs)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        X = self.inputs[index]
        return X


class Network(nn.Module):
    def __init__(self, all_X_train, random_state=42):
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        super(Network, self).__init__()
        self.all_X_train = all_X_train

        self.output_criterion = nn.BCELoss()
        self.optimizer = None

    def __initialize_network(self, input_shape, output_shape):
        self.layer1 = nn.Linear(input_shape, 64)
        self.reconstruct_output = nn.Linear(64, input_shape)

        self.encode_layer = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        self.decode_layer = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_shape)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(16, 16),
            nn.Linear(16, output_shape),
        )
        self.optimizer = Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        encoded_feature = self.encode_layer(inputs)

        # output
        output = torch.sigmoid(self.output_layer(encoded_feature))

        # reconstruction
        reconstruct_output = self.decode_layer(encoded_feature)

        return output, reconstruct_output

    def freeze_encoder_layer(self):
        for layer in list(self.encode_layer.parameters()):
            layer.requires_grad = False

    def unfreeze_encoder_layer(self):
        for layer in list(self.encode_layer.parameters()):
            layer.requires_grad = True

    def fit(self, limited_inputs, limited_targets):
        # hard code the value 1 for now, we are only predicting 2 values
        self.__initialize_network(limited_inputs.shape[1], 1)
        self.train(True)
        # make dataset to torch type
        dataset = OnlyXDataset(self.all_X_train)
        # shuffle false because data already shuffled
        data_loader = DataLoader(dataset, batch_size, shuffle=False)
        limited_dataset = TorchDataset(limited_inputs, limited_targets)
        # shuffle false because data already shuffled
        limited_data_loader = DataLoader(limited_dataset, batch_size, shuffle=False)

        self.unfreeze_encoder_layer()
        # train autoencoder first
        for j in range(epoch):
            for all_x in data_loader:
                variable_all_x = Variable(all_x)
                _, reconstruction = self.forward(variable_all_x)
                reconstruction_loss = torch.abs(reconstruction - variable_all_x).mean()
                self.optimizer.zero_grad()
                reconstruction_loss.backward()
                self.optimizer.step()

        # after training autoencoder freeze the previous layers
        self.freeze_encoder_layer()

        for i in range(epoch):
            for batch_limit_x, batch_limit_y in limited_data_loader:
                variable_batch_limit_x = Variable(batch_limit_x)
                output, _ = self.forward(variable_batch_limit_x)
                output_loss = self.output_criterion(output, batch_limit_y)

                # batch_all_x, _ = next(iter(data_loader))
                # variable_batch_all_x = Variable(batch_all_x)
                # _, reconstruction = self.forward(batch_all_x)
                # reconstruction_loss = torch.abs(reconstruction - variable_batch_all_x).mean()

                total_loss = output_loss #+ 0.5 * reconstruction_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # print(f"output loss: {output_loss.item()}")
                # print(f"reconstruction loss: {reconstruction_loss.item()}")

    def score(self, inputs, targets):
        self.train(False)
        self.eval()

        test_dataset = TorchDataset(inputs, targets)
        data_loader = DataLoader(test_dataset, batch_size, shuffle=False)  # shuffle false because data already shuffled

        for batch_x, batch_y in data_loader:
            variable_batch_x = Variable(batch_x)
            variable_batch_y = Variable(batch_y)

            output, reconstruction = self.forward(variable_batch_x)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            accuracy = (output.shape[0] - torch.abs(output - variable_batch_y).sum()) / output.shape[0]

            return accuracy