from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, f1_score
from scipy.stats import friedmanchisquare
import statsmodels.stats.api as sms
import scipy.stats
import scikit_posthocs as sp
from sklearn import metrics
import re
import math

from joblib import dump, load
import numpy as np
import pandas as pd
import seaborn as sns

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
import matplotlib.colors as mcolors

import radMLBench


def getCI(arr):
    mean = np.mean(arr)
    confidence_interval = sms.DescrStatsW(arr).tconfint_mean()
    mean_rounded = round(mean, 2)
    ci_lower_rounded = round(confidence_interval[0], 2)
    ci_upper_rounded = round(confidence_interval[1], 2)
    ci_str = f"{ci_lower_rounded} - {ci_upper_rounded}"
    return f"{mean_rounded} ({ci_str})"


def addBorder (img, pos, thickness):
    if pos == "H":
        img = np.hstack([255*np.ones(( img.shape[0],int(img.shape[1]*thickness), 3), dtype = np.uint8),img])
    if pos == "V":
        img = np.vstack([255*np.ones(( int(img.shape[0]*thickness), img.shape[1], 3), dtype = np.uint8),img])
    return img


def extract_main_value(value):
    match = re.match(r'([+-]?[0-9]*\.?[0-9]+)', value)
    if match:
        return float(match.group(1))
    return None


def getMLStr(cr):
    f = cr["params_N"]
    clf = cr["params_clf_method"]
    fs = cr["params_fs_method"]
    if clf == "RBFSVM":
        C = math.log2(cr['params_C_SVM'])
        clf_p = f" (C={int(C)})"
    elif clf == "LogisticRegression":
        C = math.log2(cr['params_C_LR'])
        clf_p = f" (C={int(C)})"
    elif clf == "RandomForest":
        C = cr["params_RF_n_estimators"]
        clf_p = f" (N={int(C)})"
    else:
        clf_p = '' # naive bayes
    mlstr = f"{fs} (f={f})\n{clf}{clf_p}"
    return mlstr


def createPlots (df, metric):
    # now we have the best models and can compute all kind of things
    array = np.array
    n_splits = df.iloc[0]["user_attrs_num_splits"]
    n_repeats = df.iloc[0]["user_attrs_num_repeats"]
    ftbl = pd.DataFrame(index=radMLBench.listDatasets(), columns=["1.0", "0.95", "0.9", "0.8"])
    for dataset in radMLBench.listDatasets():
        subdf = df.query("dataset == @dataset").reset_index(drop = True)
        assert (len(subdf) == 4)
        tblrow = {}
        for idx in range(len(subdf)):
            cr = subdf.iloc[idx]
            gts = eval(cr["user_attrs_y_gt"])
            preds = eval(cr["user_attrs_y_probs"])
            stats = []
            for r in range(n_repeats):
                gt = np.concatenate([a for a in gts[r*n_splits:(r+1)*n_splits]])
                pred = np.concatenate([a for a in preds[r*n_splits:(r+1)*n_splits]])
                auc = roc_auc_score(gt, pred)
                sens = recall_score(gt, pred.round(), pos_label=1)
                spec = recall_score(gt, pred.round(), pos_label=0)
                if metric == "auc":
                    stats.append(auc)
                elif metric == "sens":
                    stats.append(sens)
                elif metric == "spec":
                    stats.append(spec)
                elif metric == "ML":
                    pass
                else:
                    raise Exception ("Unknown metric")
            thr = cr["Threshold"]
            if metric == "ML":
                ftbl.at[dataset, str(thr)] = getMLStr(cr)
            else:
                tblrow[f"{thr}"] = stats
        if metric == "ML":
            pass
        else:
            ftbl.at[dataset, "1.0"] = getCI(tblrow["1.0"])
            for ts in ["0.95", "0.9", "0.8"]:
                diffs = np.array(tblrow[ts]) - np.array(tblrow["1.0"])
                ftbl.at[dataset, ts] = getCI(diffs)

    def beautify(value):
        match = re.match(r'([+-]?[0-9]*\.?[0-9]+)\s+\(([^\)]+)\)', value)
        if match:
            return f"{match.group(1)}\n({match.group(2)})"
        return None

    if metric == "ML":
        def place_1(value):
            return 0.0
        dfheat = ftbl.applymap(place_1)
        maxx = 1.0
    else:
        dfheat = ftbl.applymap(extract_main_value)
        minV = np.min(dfheat.iloc[:,1:4])
        maxV = np.max(dfheat.iloc[:,1:4])
        maxx = np.max([np.abs(minV), np.abs(maxV)])
    norm = TwoSlopeNorm(vmin=-maxx, vcenter=0, vmax=maxx)
    # cmap = sns.diverging_palette(h_neg=40, h_pos=260, s=90, l=70, as_cmap=True)
    if metric == "ML":
        colors = [(0, 'white'), (0.5, 'white'), (1.0, 'white')]
    else:
        colors = [(0, 'orange'), (0.5, 'white'), (1.0, 'skyblue')]
    cmap = LinearSegmentedColormap.from_list('custom', colors)

    tbl3 = dfheat.round(3).applymap(lambda x: 0.0 if abs(x) < 0.001 else x)
    if metric == "ML":
        ftblb = ftbl.copy()
    else:
        ftblb = ftbl.applymap(beautify)
    w = 7.5
    if metric == "ML":
        w = 13
    createPlot(dfheat.iloc[0:25], ftblb.iloc[0:25], cmap, norm, f"./paper/Figure_S1_{metric}_A.png", w, 13.0)
    createPlot(dfheat.iloc[25:], ftblb.iloc[25:], cmap, norm, f"./paper/Figure_S1_{metric}_B.png", w, 13.0)




def createPlot(tbl, ftbl, cmap, norm, fname, w, h):
    plt.figure(figsize=(w, h))

    mask = np.zeros_like(tbl)
    mask[:, 0] = True
    ax = sns.heatmap(tbl, annot=ftbl, mask=mask, fmt='', norm=norm, cmap=cmap, linewidths=0.5, linecolor='black', cbar=False)
    for _, spine in ax.spines.items():
        spine.set_visible(True)

    # copy text..
    for j, k in enumerate(ftbl.iloc[:,0]):
        ax.text(0.5, 0.5+j, k, fontsize=10, ha="center", va="center")

    plt.title('')
    plt.xlabel('Decorrelation threshold')
    plt.ylabel('')
    plt.tick_params(axis='y', which='both', length=0)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)



def generateSummaryPlot (df):
    #thresholds = [0.8, 0.9, 0.95, 1.0]
    thresholds = [1.0, 0.95, 0.9, 0.8]
    cumulative_df = df.reset_index()
    df_melted = cumulative_df.reset_index().melt(id_vars='index', value_vars=thresholds, var_name='Threshold', value_name='Value')
    df_melted.rename(columns={'index': 'Dataset'}, inplace=True)
    df_melted['Threshold'] = pd.Categorical(df_melted['Threshold'], categories=thresholds, ordered=True)
    mean_values = df.mean()
    #threshold_positions = [0, 1, 2, 3]
    threshold_positions = [3, 2, 1, 0]
    colors = {1.0: '#C0C0C0',  0.95: '#A0A0A0',  0.9: '#808080', 0.8: '#606060'}

    plt.figure(figsize=(12, 10))
    sns.boxplot(x='Threshold', y='Value', data=df_melted, hue='Threshold', palette=colors, showfliers=False, legend=False, width=0.4)


    slopes = {}
    for dataset in cumulative_df.index:
        y_vals = cumulative_df.loc[dataset, thresholds].values
        x_vals = thresholds
        slopes_list = [(y_vals[i+1] - y_vals[i]) / (x_vals[i+1] - x_vals[i]) for i in range(len(thresholds)-1)]
        mean_slope = sum(slopes_list) / len(slopes_list)
        slopes[dataset] = mean_slope

    vabsmax = max(abs(min(slopes.values())), abs(max(slopes.values())))
    cmap = mcolors.LinearSegmentedColormap.from_list('SlopeMap', ['blue', 'white', 'orange'])
    norm = mcolors.Normalize(vmin=-vabsmax, vmax=vabsmax)
    for dataset in cumulative_df.index:
        slope = slopes[dataset]
        color = cmap(norm(slope))
        plt.plot(threshold_positions, cumulative_df.loc[dataset, thresholds[::-1]],
                 marker='o', color=color, alpha=0.6, linestyle='-', linewidth=1.5)


    mean_line = plt.plot(threshold_positions, mean_values[thresholds[::-1]],
                     marker='o', color='black', linestyle='--', linewidth=2.5, label='Mean')

    plt.xlabel('Decorrelation threshold', fontsize=24)
    plt.ylabel('AUC', fontsize=24)
    # plt.title('Effect of decorrelation thresholds on AUC', fontsize=24)
    plt.xticks(ticks=threshold_positions, labels=[str(t) for t in thresholds[::-1]], fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend( loc='upper left', fontsize=24)
    plt.savefig('./paper/Figure3.png', dpi=300, bbox_inches='tight')



def getResults ():
    array = np.array
    results = []
    for dataset in radMLBench.listDatasets():
        X, y = radMLBench.loadData(dataset, return_X_y = True, local_cache_dir = "./datasets")
        for th in [0.8, 0.9, 0.95, 1.0]:
            z = f"./examples/decorrelation/results/trial_{dataset}_{th}.csv"
            df = pd.read_csv(z, compression = "gzip")
            n_splits = df.iloc[0]["user_attrs_num_splits"]
            n_repeats = df.iloc[0]["user_attrs_num_repeats"]

            for idx in range(len(df)):
                cr = df.iloc[idx]
                gts = eval(cr["user_attrs_y_gt"])
                preds = eval(cr["user_attrs_y_probs"])
                a_auc = [];
                for r in range(n_repeats):
                    gt = np.concatenate([a for a in gts[r*n_splits:(r+1)*n_splits]])
                    pred = np.concatenate([a for a in preds[r*n_splits:(r+1)*n_splits]])
                    auc = roc_auc_score(gt, pred)
                    a_auc.append(auc)
                mean_auc = np.mean(a_auc)
                ci_auc = 1.96 * np.std(a_auc) / np.sqrt(n_repeats)
                df.at[idx, "mAUC"] = mean_auc
                df.at[idx, "mAUC_CI"] = ci_auc
            br = df.sort_values(["mAUC"]).iloc[-1].copy()
            br["dataset"] = dataset
            br["threshold"] = th
            results.append(br)
    results = pd.DataFrame(results)
    return results


if __name__ == '__main__':
    #results = load("./examples/decorrelation/results/results.dump")
    try:
        results = load("./examples/decorrelation/results/results_trial.dump")
    except:
        print("Recomputing AUCs")
        results = getResults () # recompute everything
        _ = dump(results, "./examples/decorrelation/results/results_trial.dump")

    results=pd.DataFrame(results).reset_index(drop = True)
    results["auc"] = results["mAUC"]

    df = results[['dataset', 'threshold', 'auc']]
    df = df.pivot(index='dataset', columns='threshold', values='auc')
    ranks = df.rank(axis=1, ascending=False, method='average').mean(axis = 0)
    print (ranks)

    # test
    K = df.reset_index(drop = True)
    s, p = friedmanchisquare(*K.values)
    print ("Friedman test:", p)
    nemenyi_results = sp.posthoc_nemenyi_friedman(K.values)
    print(nemenyi_results)

    # just print
    for threshold in df.columns:
        mean = round(df[threshold].mean(), 3)
        std = round(df[threshold].std(), 3)
        print(f"Threshold {threshold}: mean {mean} +/- std {std}")


    createPlots(results, "auc")
    createPlots(results, "sens")
    createPlots(results, "spec")
    createPlots(results, "ML")
    generateSummaryPlot (df)




#
