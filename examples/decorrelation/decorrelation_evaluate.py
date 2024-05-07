from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import ttest_rel
from scipy.stats import friedmanchisquare

import cv2
import joblib
from joblib import dump, load
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

import radMLBench




def addBorder (img, pos, thickness):
    if pos == "H":
        img = np.hstack([255*np.ones(( img.shape[0],int(img.shape[1]*thickness), 3), dtype = np.uint8),img])
    if pos == "V":
        img = np.vstack([255*np.ones(( int(img.shape[0]*thickness), img.shape[1], 3), dtype = np.uint8),img])
    return img


def join_plots():
    fontFace = "Arial"

    imA = cv2.imread("./paper/Figure_3A.png")
    imB = cv2.imread("./paper/Figure_3B.png")
    imB = addBorder (imB, "H", 0.025)
    imgU = np.hstack([imA, imB])
    cv2.imwrite("./paper/Figure_3.png", imgU)


def createPlot(tbl, cmap, fname, w):
    plt.figure(figsize=(w, 7.7))
    ax = sns.heatmap(tbl, annot=True, cmap=cmap, linewidths=0.5, linecolor='black', cbar=False)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    plt.title('')
    plt.xlabel('Decorrelation threshold')
    plt.ylabel('')
    plt.tick_params(axis='y', which='both', length=0)
    plt.tight_layout()
    plt.savefig(fname, dpi=400)


if __name__ == '__main__':
    results = load("./examples/decorrelation/results/results.dump")

    df = pd.DataFrame(results, columns=['dataset', 'threshold', 'auc'])
    df = df.pivot(index='dataset', columns='threshold', values='auc')
    ranks = df.rank(axis=1, ascending=False, method='average').mean(axis = 0)
    print (ranks)
    s, p = friedmanchisquare(*[df[col] for col in df.columns])
    print ("Friedman test:", p)

    for threshold in df.columns:
        mean = round(df[threshold].mean(), 3)
        std = round(df[threshold].std(), 3)
        print(f"Threshold {threshold}: mean {mean} +/- std {std}")



    tbl3 = df.sub(df[1.0], axis=0)
    minV = np.min(tbl3)
    maxV = np.max(tbl3)
    tbl3[1.0] = df[1.0]
    tbl3 = tbl3.round(3).applymap(lambda x: 0.0 if abs(x) < 0.001 else x)
    colors = [(0, 'chocolate'), (-minV, 'white'), (maxV-minV, 'green'), (maxV-minV+0.1, 'white'), (1, 'white')]
    cmap = LinearSegmentedColormap.from_list('custom', colors)

    createPlot(tbl3.iloc[0:25], cmap, "./paper/Figure_3A.png", 4.9)
    createPlot(tbl3.iloc[25:],cmap, "./paper/Figure_3B.png", 4.6)
    join_plots()



#
