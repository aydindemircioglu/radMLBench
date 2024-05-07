#

from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel, f_classif, SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from tabpfn import TabPFNClassifier

import cv2
import joblib
from joblib import dump, load
import numpy as np
import os
import pandas as pd
import random

import radMLBench

import optuna
import warnings
from optuna.exceptions import ExperimentalWarning
optuna.logging.set_verbosity(optuna.logging.FATAL)



search_space_A = {
    'fs_method': ["Bhattacharyya", "ANOVA", "LASSO", "ET"],
    'N': [2**k for k in range(0,7)],
    'clf_method': ["RBFSVM", "RandomForest", "LogisticRegression", "NaiveBayes"],
    'RF_n_estimators': [10,25,50,100,250,500,1000],
    'C_LR': [2**k for k in range(-7,7,2)],
    'C_SVM': [2**k for k in range(-7,7,2)]
}

search_space_B = {
    'fs_method': ["Bhattacharyya", "ANOVA", "LASSO", "ET"],
    'N': [64],
    'clf_method': ["TabPFN"],
}



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



def objective(trial, dataset, search_space):
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
    if clf_method == "TabPFN":
        clf = TabPFNClassifier(device='cuda', N_ensemble_configurations = 32)

    X, y = radMLBench.loadData(dataset, return_X_y = True, local_cache_dir = "./datasets")
    cvSplits = radMLBench.getCVSplits((X,y), num_splits=10, num_repeats=1)

    y_probs = []
    y_gt = []
    for i, (train_index, test_index) in enumerate(cvSplits):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        nFeatures_corr = min(N, X_train.shape[1])
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

        X_train_selected = fsel.fit_transform(X_train, y_train)
        X_test_selected = fsel.transform(X_test)

        clf.fit(X_train_selected, y_train)
        y_prob = clf.predict_proba(X_test_selected)[:, 1]
        y_probs.extend(y_prob)
        y_gt.extend(y_test)

    cv_auc = roc_auc_score(y_gt, y_probs)
    return cv_auc



def optimize_dataset_threshold(dataset, search_space, expName):
    print("Starting dataset:", dataset, "on experiment:", expName)
#    optuna.logging.set_verbosity(optuna.logging.ERROR)
    warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.samplers")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler(), direction="maximize")
        study.optimize(lambda trial: objective(trial, dataset, search_space))
        best_params = study.best_params
        auc = study.best_value
        df = study.trials_dataframe()
        print("\tOptimizing for dataset:", dataset, "on experiment:", expName, "obtained", auc, "  #Trials", len(df))
    return dataset, auc



datasets = radMLBench.listDatasets("nInstances")[::-1] # small datasets last

# print warning
print ("WARNING: In case tabPFN was not yet used, this skript will stall, since multiple")
print ("instances will try to download the model at the same time. In this case, please")
print ("set downloadFirst in this training first to True, then execute, then set to False.")

downloadFirst = False
if downloadFirst == True:
    # ensure that the model was downloaded
    optimize_dataset_threshold(datasets[0], search_space_B, "TabPFN")
    exit(-1)


# standard gridsearch
results = joblib.Parallel(n_jobs=30)(
    joblib.delayed(optimize_dataset_threshold)(dataset, search_space_A, "Standard")
    for dataset in datasets
)
dump(results, "./examples/tabpfn/results/results_Standard.dump")

# tabpfn
for nFeatures in [1,2,4,8,16,32,64,100]:
    search_space_B["N"] = [nFeatures]
    results = joblib.Parallel(n_jobs=4)(
        joblib.delayed(optimize_dataset_threshold)(dataset, search_space_B, "TabPFN")
        for dataset in datasets
    )
    dump(results, f"./examples/tabpfn/results/results_TabPFN_{nFeatures}.dump")


#
