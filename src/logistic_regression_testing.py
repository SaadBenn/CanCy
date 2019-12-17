import argparse
import numpy as np
import os
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
import sys

sys.path.insert(0, "./")

from src.data_preprocessing import dataset_dict
from src.network.network import Network, NetworkSecondApproach, NetworkThirdApproach, NetworkFourthApproach

def limit_samples(inputs, targets, num_classes=2, num_samples_per_class=5):
    """
    Limit the number of samples per class

    :param inputs:
    :param targets:
    :param num_classes:             The total number of classes
    :param num_samples_per_class:   Number of samples to train for model for each class
    :return:
    """
    limited_inputs = []
    limited_targets = []

    targets_count = {}
    total_samples = num_classes * num_samples_per_class
    num_samples_added = 0

    # loop through the dataset and add the samples for each class to targets_dict
    for index in range(len(targets)):
        current_target = targets[index][0]
        if targets_count.get(current_target, None) is None:
            targets_count[current_target] = 1
            limited_inputs.append(inputs[index])
            limited_targets.append(targets[index])
            num_samples_added += 1
        else:
            # if the length of this current target is less than the num_samples_per_class we append
            if targets_count[current_target] < num_samples_per_class:
                targets_count[current_target] += 1
                limited_inputs.append(inputs[index])
                limited_targets.append(targets[index])
                num_samples_added += 1
                # check if the total targets_dict reach the limit
                if len(limited_inputs) >= total_samples:
                    break

    limited_inputs = np.array(limited_inputs)
    limited_targets = np.array(limited_targets)
    return limited_inputs, limited_targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", default=4, help="Set the random state so we can reproduce the result")
    parser.add_argument('--dataset_type', required=True, help="automatically looks into data directory"
                                                              "example would be breast-cancer-wisconsin.data")
    parser.add_argument('--model_type', required=True, help="choose the model to train on the dataset")
    parser.add_argument('--num_samples', type=int, help="specify the number of samples for training the model")
    parser.add_argument('--num_folds', type=int, default=5, help="specify the number of samples for training the model")
    parser.add_argument('--train_size', type=float, default=0.8, help="specify the percentage of data (0-1) to be used for training")
    arg = parser.parse_args()

    # get dataset
    dataset_filename = os.path.join('data', arg.dataset_type)
    dataset = dataset_dict[arg.dataset_type](dataset_filename, seed=arg.random_seed, num_kfold_splits=arg.num_folds)
    dataset.process_dataset()
    dataset.split_into_train_and_test(arg.train_size)

    param_distributions ={
    'l1_ratio' : stats.uniform(0, 1),
    'C' : loguniform(1e-4, 1e0),
    }

    model = LogisticRegression(penalty='elasticnet', solver = 'saga', random_state=arg.random_seed, max_iter=10000)

    best_model_found = rand_search.fit(dataset.X, np.ravel(dataset.Y))
    print("f1 score on training: ", best_model_found.best_score_)
    
    prediction = best_model_found.best_estimator_.predict(dataset.X_Test)
    score = f1_score(dataset.Y_Test, prediction, average='micro')
    print("f1 score on testing: ", score)

