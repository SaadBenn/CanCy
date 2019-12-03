import argparse
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
    if arg.num_samples:
        X_train = X_train[:arg.num_samples]
        y_train = y_train[:arg.num_samples]

    # train model
    model_initializer = model_dict[arg.model_type]
    model = model_initializer(random_state=arg.random_seed)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Accuracy: {score}")
