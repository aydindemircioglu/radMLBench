
import cv2
from joblib import dump, load
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

import sys
sys.path.append("./")
import radMLBench




def addBorder (img, pos, thickness):
    if pos == "H":
        img = np.hstack([255*np.ones(( img.shape[0],int(img.shape[1]*thickness), 3), dtype = np.uint8),img])
    if pos == "V":
        img = np.vstack([255*np.ones(( int(img.shape[0]*thickness), img.shape[1], 3), dtype = np.uint8),img])
    return img


def join_plots():
    imA = cv2.imread("./paper/Figure_4A.png")
    imB = cv2.imread("./paper/Figure_4B.png")
    imB = addBorder (imB, "H", 0.025)
    imgU = np.hstack([imA, imB])
    cv2.imwrite("./paper/Figure_4.png", imgU)


def createPlot(tbl, cmap, fname, w):
    plt.figure(figsize=(w, 7.7))
    ax = sns.heatmap(tbl, annot=True, cmap=cmap, linewidths=0.5, linecolor='black', cbar=False)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    plt.tick_params(axis='y', which='both', length=0)
    plt.tight_layout()
    plt.savefig(fname, dpi=400)


if __name__ == '__main__':
    resStd = load("./examples/tabpfn/results/results_Standard.dump")
    dfStd = pd.DataFrame(resStd, columns=['dataset', 'auc'])

    for nFeatures in [1,2,4,8,16,32,64, 100]:
        resTabPFN = load(f"./examples/tabpfn/results/results_TabPFN_{nFeatures}.dump")
        dfTabPFN = pd.DataFrame(resTabPFN, columns=['dataset', 'auc'])

        # merge both
        df = dfTabPFN.merge(dfStd, on = "dataset")
        dfd = df.rename(columns={'auc_x': 'TabPFN', 'auc_y': 'Standard'})
        df = dfd.drop(["dataset"], axis = 1)
        # show ranks
        ranks = df.rank(axis=1, ascending=False, method='average').mean(axis = 0)
        print ("\n", nFeatures)
        print (ranks)

        # then apply friedman test
        s, p = wilcoxon(*[df[col] for col in df.columns])
        print ("Wilcoxon test:", p)

        for method in df.columns:
            mean = round(df[method].mean(), 3)
            std = round(df[method].std(), 3)
            print(f"{method}_{nFeatures}: mean {mean} +/- std {std}")

    # plot differences somehow. standard is base, TabPFN is difference
    # we only plot N=100, which has the best rank.
    # after the loop above, we have N=100 model already in df
    tbl3 = df.sub(df["Standard"], axis=0)
    minV = np.min(tbl3)
    maxV = np.max(tbl3)
    tbl3["Standard"] = df["Standard"]
    tbl3.index = dfd["dataset"]
    tbl3 = tbl3.sort_values(["dataset"]).copy()

    tbl3 = tbl3.round(3).applymap(lambda x: 0.0 if abs(x) < 0.001 else x)
    colors = [(0, 'chocolate'), (-minV, 'white'), (maxV-minV, 'green'), (maxV-minV+0.1, 'white'), (1, 'white')]
    cmap = LinearSegmentedColormap.from_list('custom', colors)

    createPlot(tbl3.iloc[0:25], cmap, "./paper/Figure_4A.png", 3.8)
    createPlot(tbl3.iloc[25:],cmap, "./paper/Figure_4B.png", 3.6)
    join_plots()



#
