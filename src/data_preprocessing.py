import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split


class Dataset:
    def __init__(self, path_to_file, seed=42, num_kfold_splits=5):
        self.filename = path_to_file
        self._X = None
        self._X_Test = None
        self._Y = None
        self._Y_Test = None
        self._kfold_dataset = None
        self.seed = seed
        self.scale = StandardScaler()
        # kfold
        self.num_kfold_splits = num_kfold_splits
        self.current_kfold_index = 0

    @property
    def X(self):
        return self._X
    
    @property
    def X_Test(self):
        return self._X_Test

    @property
    def Y(self):
        return self._Y
    
    @property
    def Y_Test(self):
        return self._Y_Test

    @property
    def kfold_dataset(self):
        return self._kfold_dataset

    def process_dataset(self):
        # normalize X
        self._X = self.scale.fit_transform(self._X)
        
        # using Label Encoder to convert categorical data into number so the
        # model can understand better
        labelencoder_Y = LabelEncoder()
        self._Y = labelencoder_Y.fit_transform(self._Y)
        self._Y = np.expand_dims(self._Y, 1).astype(np.float32)

    def create_kfold_dataset(self):
        kfold = KFold(self.num_kfold_splits, shuffle=True, random_state=self.seed)
        self._kfold_dataset = list(kfold.split(self._X, self._Y))  # get the index

    def get_next_kfold_data(self):
        current_kfold_data = self._kfold_dataset[self.current_kfold_index]
        self.current_kfold_index += 1
        # goes back if current fold is more than the number of folds
        if self.current_kfold_index >= self.num_kfold_splits:
            self.current_kfold_index = 0

        current_X_train = self._X[current_kfold_data[0]]
        current_y_train = self._Y[current_kfold_data[0]]
        current_X_test = self._X[current_kfold_data[1]]
        current_y_test = self._Y[current_kfold_data[1]]

        return current_X_train, current_X_test, current_y_train, current_y_test
    
    def split_into_train_and_test(self, train_size):
        self._X, self._X_Test, self._Y, self._Y_Test = train_test_split(self._X, self._Y, train_size=train_size,
                                                                         shuffle=True, random_state=self.seed)

class BCWDataset(Dataset):
    def __init__(self, path_to_file, seed, num_kfold_splits=5):
        super(BCWDataset, self).__init__(path_to_file, seed, num_kfold_splits=5)

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
    def __init__(self, path_to_file, seed, num_kfold_splits=5):
        super(WDBCDataset, self).__init__(path_to_file, seed, num_kfold_splits=5)

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
    def __init__(self, path_to_file, seed, num_kfold_splits=5):
        super(WPBCDataset, self).__init__(path_to_file, seed, num_kfold_splits=5)

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

