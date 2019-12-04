import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, path_to_file, seed=42):
        self.filename = path_to_file
        self._X = None
        self._Y = None
        self.seed = seed
        self.scale = StandardScaler()

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y
    
    def process_dataset(self):
        # normalize X
        self._X = self.scale.fit_transform(self._X)
        
        # using Label Encoder to convert categorical data into number so the
        # model can understand better
        labelencoder_Y = LabelEncoder()
        self._Y = labelencoder_Y.fit_transform(self._Y)
        self._Y = np.expand_dims(self._Y, 1)

    def create_train_test_dataset(self):
        X_train, X_test, y_train, y_test = train_test_split(self._X, self._Y, test_size=0.2,
                                                            shuffle=True, random_state=self.seed)
        return X_train, X_test, y_train, y_test


class BCWDataset(Dataset):
    def __init__(self, path_to_file, seed):
        super(BCWDataset, self).__init__(path_to_file, seed)

    def process_dataset(self):
        # Reading the data set
        data = pd.read_csv(self.filename)
        X = data.iloc[:, 1:10].values
        Y = data.iloc[:, 10].values
        # replace missing value with 0
        X[np.where(X == '?')] = 0
        X = X.astype(np.float32)
        self._X = X
        self._Y = Y


        print("breast cancer wisconsin cancer dataset dimensions : {}".format(data.shape))
        super(BCWDataset, self).process_dataset()


class WDBCDataset(Dataset):
    def __init__(self, path_to_file, seed):
        super(WDBCDataset, self).__init__(path_to_file, seed)

    def process_dataset(self):
        # Reading the data set
        data = pd.read_csv(self.filename)
        X = data.iloc[:, 2:].values
        Y = data.iloc[:, 1].values
        Y[np.where(Y == 'B')] = 0
        Y[np.where(Y == 'M')] = 1
        X = X.astype(np.float32)
        self._X = X
        self._Y = Y

        print("WDBC cancer dataset dimensions : {}".format(data.shape))
        super(WDBCDataset, self).process_dataset()


class WPBCDataset(Dataset):
    def __init__(self, path_to_file, seed):
        super(WPBCDataset, self).__init__(path_to_file, seed)

    def process_dataset(self):
        # Reading the data set
        data = pd.read_csv(self.filename)
        X = data.iloc[:, 2:].values
        Y = data.iloc[:, 1].values
        X[np.where(X == '?')] = 0
        Y[np.where(Y == 'N')] = 0
        Y[np.where(Y == 'R')] = 1
        X = X.astype(np.float32)
        self._X = X
        self._Y = Y

        print("WPBC cancer dataset dimensions : {}".format(data.shape))
        super(WPBCDataset, self).process_dataset()


dataset_dict = {
    'breast-cancer-wisconsin.data': BCWDataset,
    'wdbc.data': WDBCDataset,
    'wpbc.data': WPBCDataset
}

