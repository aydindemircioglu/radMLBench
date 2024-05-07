from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import radMLBench


for dataset in radMLBench.listDatasets():
    print ("Loading", dataset)
    X, y = radMLBench.loadData(dataset, return_X_y = True, local_cache_dir = "./datasets")
    cvSplits = radMLBench.getCVSplits((X,y), num_splits=10, num_repeats=2)
    rf = RandomForestClassifier()

    gt_labels = []
    preds = []
    for i, (train_index, test_index) in enumerate(cvSplits):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        gt_labels.extend(y_test)
        rf.fit(X_train, y_train)
        y_prob_rf = rf.predict_proba(X_test)[:, 1]
        preds.extend(y_prob_rf)

    mAUC = roc_auc_score(gt_labels, preds)
    print(f"\tAUC:", mAUC)


 #
