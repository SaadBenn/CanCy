import argparse
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
import sys

sys.path.insert(0, "./")

from src.data_preprocessing import dataset_dict
from src.network.network import Network


model_dict = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "decision_tree": DecisionTreeClassifier,
    "gradient_boost": GradientBoostingClassifier,
    "svc": SVC,
    "linear_svc": LinearSVC,
    "network": Network
}


def model_initializer(all_X_train, all_y_train, model_type, random_state=42):
    if arg.model_type == "network":
        model_selected = model_dict[model_type](all_X_train, all_y_train, random_state=random_state)
    else:
        model_selected = model_dict[model_type](random_state=random_state)
    return model_selected


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
    arg = parser.parse_args()

    # get dataset
    dataset_filename = os.path.join('data', arg.dataset_type)
    dataset = dataset_dict[arg.dataset_type](dataset_filename, seed=arg.random_seed)
    dataset.process_dataset()
    X_train, X_test, y_train, y_test = dataset.create_train_test_dataset()
    # if num_samples is specify, then we limit the training samples
    limit_X_train = X_train
    limit_y_train = y_train
    if arg.num_samples:
        limit_X_train, limit_y_train = limit_samples(X_train, y_train, num_classes=2, num_samples_per_class=arg.num_samples)

    # train model
    model = model_initializer(X_train, y_train, arg.model_type, random_state=arg.random_seed)
    model.fit(limit_X_train, limit_y_train)
    score = model.score(X_test, y_test)
    print(f"Accuracy: {score}")
