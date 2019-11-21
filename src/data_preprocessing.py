from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Dataset:
    def __init__(self, path_to_file):
        self.filename = path_to_file
        self.X = None
        self.Y = None
    
    def process_dataset(self):
        # using Label Encoder to convert categorical data into number so the
        # model can understand better
        labelencoder_Y = LabelEncoder()
        self.Y = labelencoder_Y.fit_transform(self.Y)


class BCWDataset(Dataset):
    def __init__(self, path_to_file):
        super(BCWDataset, self).__init__(path_to_file)

    def process_dataset(self):
        # Reading the data set
        data = pd.read_csv(self.filename)
        X = data.iloc[:, 1:10].values
        Y = data.iloc[:, 10].values
        # replace missing value with 0
        X[np.where(X == '?')] = 0
        X = X.astype(np.int32)
        self.X = X
        self.Y = Y

        print("breast cancer wisconsin cancer dataset dimensions : {}".format(data.shape))
        super(BCWDataset, self).process_dataset()


class WDBCDataset(Dataset):
    def __init__(self, path_to_file):
        super(WDBCDataset, self).__init__(path_to_file)

    def process_dataset(self):
        # Reading the data set
        data = pd.read_csv(self.filename)
        X = data.iloc[:, 2:].values
        Y = data.iloc[:, 1].values
        Y[np.where(Y == 'B')] = 0
        Y[np.where(Y == 'M')] = 1
        X = X.astype(np.float32)
        self.X = X
        self.Y = Y

        print("WDBC cancer dataset dimensions : {}".format(data.shape))
        super(WDBCDataset, self).process_dataset()


class WPBCDataset(Dataset):
    def __init__(self, path_to_file):
        super(WPBCDataset, self).__init__(path_to_file)
        
    def process_dataset(self):
        # Reading the data set
        data = pd.read_csv(self.filename)
        X = data.iloc[:, 2:].values
        Y = data.iloc[:, 1].values
        X[np.where(X == '?')] = 0
        Y[np.where(Y == 'N')] = 0
        Y[np.where(Y == 'R')] = 1
        X = X.astype(np.float32)
        self.X = X
        self.Y = Y

        print("WPBC cancer dataset dimensions : {}".format(data.shape))
        super(WPBCDataset, self).process_dataset()