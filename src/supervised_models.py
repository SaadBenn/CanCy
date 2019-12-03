import argparse
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC

from src.data_preprocessing import dataset_dict


model_dict = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "decision_tree": DecisionTreeClassifier,
    "gradient_boost": GradientBoostingClassifier,
    "svc": SVC,
    "linear_svc": LinearSVC
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", default=42, help="Set the random state so we can reproduce the result")
    parser.add_argument('--dataset_type', required=True, help="automatically looks into data directory"
                                                              "example would be breast-cancer-wisconsin.data")
    parser.add_argument('--model_type', required=True, help="choose the model to train on the dataset")
    arg = parser.parse_args()

    # get dataset
    dataset_filename = os.path.join('data', arg.dataset_type)
    dataset = dataset_dict[arg.dataset_type](dataset_filename, seed=arg.random_seed)
    dataset.process_dataset()
    X_train, X_test, y_train, y_test = dataset.create_train_test_dataset()

    # train model
    model_initializer = model_dict[arg.model_type]
    model = model_initializer(random_state=arg.random_seed)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)



