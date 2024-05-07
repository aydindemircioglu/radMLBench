#

from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel, f_classif, SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import cv2
import joblib
from joblib import dump, load
import numpy as np
import os
import pandas as pd
import random
import sys
sys.path.append("./")
import radMLBench

import optuna
import warnings
from optuna.exceptions import ExperimentalWarning
optuna.logging.set_verbosity(optuna.logging.FATAL)



search_space = {
    'fs_method': ["Bhattacharyya", "ANOVA", "LASSO", "ET"],
    'N': [2**k for k in range(0,7)],
    'clf_method': ["RBFSVM", "RandomForest", "LogisticRegression", "NaiveBayes"],
    'RF_n_estimators': [10,25,50,100,250,500,1000],
    'C_LR': [2**k for k in range(-7,7,2)],
    'C_SVM': [2**k for k in range(-7,7,2)]
}



def decorrelate_features(X_train, X_test, threshold):
    if threshold == 1.0:
        return X_train, X_test

    corr_matrix = np.corrcoef(X_train, rowvar=False)
    n_features = X_train.shape[1]
    correlated_features = np.abs(corr_matrix) >= threshold
    np.fill_diagonal(correlated_features, False)

    correlated_feature_indices = np.argwhere(correlated_features)
    avg_correlation = np.mean(np.abs(corr_matrix), axis=1)
    correlated_feature_indices = correlated_feature_indices[avg_correlation[correlated_feature_indices[:, 0]] > avg_correlation[correlated_feature_indices[:, 1]]]
    features_to_keep = np.delete(np.arange(n_features), np.unique(correlated_feature_indices[:, 1]))

    return X_train[:, features_to_keep], X_test[:, features_to_keep]



def bhattacharyya_score_fct (X, y):
    yn = y/np.sum(y)
    yn = np.asarray(yn, dtype = np.float32)
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        column = X[:, j]
        if np.min(column) == np.max(column):
            scores[j] = 0
        else:
            xn = (column - np.min(column)) / (np.max(column) - np.min(column))
            xn = xn / np.sum(xn)
            xn = np.asarray(xn, dtype = np.float32)
            scores[j] = cv2.compareHist(xn, yn, cv2.HISTCMP_BHATTACHARYYA)

    scores = np.asarray(scores, dtype = np.float32)
    return -scores



def objective(trial, dataset, threshold):
    np.random.seed(42)
    random.seed(42)
    fs_method = trial.suggest_categorical("fs_method", search_space['fs_method'])
    N = trial.suggest_categorical("N", search_space['N'])

    clf_method = trial.suggest_categorical("clf_method", search_space['clf_method'])
    if clf_method == "LogisticRegression":
        C_LR = trial.suggest_categorical("C_LR", search_space['C_LR'])
        clf = LogisticRegression(max_iter=500, solver='liblinear', C = C_LR, random_state = 42)
    if clf_method == "NaiveBayes":
        clf = GaussianNB()
    if clf_method == "RandomForest":
        RF_n_estimators = trial.suggest_categorical("RF_n_estimators", search_space['RF_n_estimators'])
        clf = RandomForestClassifier(n_estimators = RF_n_estimators)
    if clf_method == "RBFSVM":
        C_SVM = trial.suggest_categorical("C_SVM", search_space['C_SVM'])
        C_gamma = 'auto'
        clf = SVC(kernel = "rbf", C = C_SVM, gamma = C_gamma, probability = True)

    X, y = radMLBench.loadData(dataset, return_X_y = True, local_cache_dir = "./datasets")
    cvSplits = radMLBench.getCVSplits((X,y), num_splits=10, num_repeats=1)

    y_probs = []
    y_gt = []
    for i, (train_index, test_index) in enumerate(cvSplits):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_decorrelated, X_test_decorrelated = decorrelate_features(X_train, X_test, threshold)

        nFeatures_corr = min(N, X_train_decorrelated.shape[1])
        if fs_method == "LASSO":
            clf_fs = LogisticRegression(penalty='l1', max_iter = 100, solver='liblinear', C = 1, random_state = 42)
            fsel = SelectFromModel(clf_fs, prefit=False, max_features=nFeatures_corr, threshold=-np.inf)
        if fs_method == "ANOVA":
            fsel = SelectKBest(f_classif, k = nFeatures_corr)
        if fs_method == "Bhattacharyya":
            fsel = SelectKBest(bhattacharyya_score_fct, k = nFeatures_corr)
        if fs_method == "MRMR":
            fsel = MRMRSelector(K=nFeatures_corr)
        if fs_method == "ET":
            clf_fs = ExtraTreesClassifier(random_state = 42)
            fsel = SelectFromModel(clf_fs, prefit=False, max_features=nFeatures_corr, threshold=-np.inf)

        X_train_selected = fsel.fit_transform(X_train_decorrelated, y_train)
        X_test_selected = fsel.transform(X_test_decorrelated)

        clf.fit(X_train_selected, y_train)
        y_prob = clf.predict_proba(X_test_selected)[:, 1]
        y_probs.extend(y_prob)
        y_gt.extend(y_test)

    cv_auc = roc_auc_score(y_gt, y_probs)
    return cv_auc



def optimize_dataset_threshold(dataset, threshold):
    print("Starting dataset:", dataset, "with threshold:", threshold)
#    optuna.logging.set_verbosity(optuna.logging.ERROR)
    warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.samplers")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler(), direction="maximize")
        study.optimize(lambda trial: objective(trial, dataset, threshold))
        best_params = study.best_params
        auc = study.best_value
        df = study.trials_dataframe()
        print("\tOptimizing for dataset:", dataset, "with threshold:", threshold, "obtained", auc, "  #Trials", len(df))
        os.makedirs("./examples/decorrelation/results", exist_ok = True)
        df.to_csv(f"./examples/decorrelation/results/trial_{dataset}_{threshold}.csv", compression = "gzip", index = False)
    return dataset, threshold, auc



datasets = radMLBench.listDatasets("nInstances")[::-1] # small datasets last
thresholds = [1.0, 0.95, 0.90, 0.80]  # 1 = no decorrelation

results = joblib.Parallel(n_jobs=30)(
    joblib.delayed(optimize_dataset_threshold)(dataset, threshold)
    for dataset in datasets
    for threshold in thresholds
)

dump(results, "./examples/decorrelation/results/results.dump")


#
