from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from utils.helpers import *


def feature_extraction(img_patches):
    X2d = np.asarray(
        [extract_features_2d(img_patches[i]) for i in range(len(img_patches))]
    )
    X6d = np.asarray(
        [extract_features(img_patches[i]) for i in range(len(img_patches))]
    )
    X8d = np.asarray([np.concatenate((X2d[i], X6d[i])) for i in range(len(X2d))])
    print(
        "Dimensions of the three different feature extraction methods (2D,6D,8D): {},{} and {}\n".format(
            X2d.shape, X6d.shape, X8d.shape
        )
    )
    return X2d, X6d, X8d


def label_extraction(gt_patches, foreground_threshold=0.25):
    y = np.asarray(
        [
            value_to_class(np.mean(gt_patches[i]), foreground_threshold=0.25)
            for i in range(len(gt_patches))
        ]
    )
    return y


def optimize_logistic_regression(X, y):
    param_grid = [
        {
            "penalty": ["l1", "l2"],
            "C": np.logspace(-6, 6, 10),
            "solver": ["liblinear"],
            "max_iter": [100, 1000],
        },
        {
            "penalty": ["l2"],
            "C": np.logspace(-6, 6, 10),
            "solver": ["lbfgs", "sag", "newton-cg"],
            "max_iter": [100, 1000],
        },
        {
            "penalty": ["elasticnet"],
            "C": np.logspace(-6, 6, 10),
            "solver": ["saga"],
            "l1_ratio": [0.1, 0.5, 0.9],
            "max_iter": [100, 1000],
        },
        {
            "penalty": ["none"],
            "solver": ["lbfgs", "sag", "newton-cg"],
            "max_iter": [100, 1000],
        },
    ]
    clf = GridSearchCV(
        LogisticRegression(class_weight="balanced"),
        param_grid=param_grid,
        cv=10,
        scoring="f1",
        verbose=1,
        n_jobs=-1,
    )

    clf.fit(X, y)

    best_model = clf.best_estimator_
    best_params = clf.best_params_
    best_f1 = clf.best_score_

    return best_model, best_params, best_f1
