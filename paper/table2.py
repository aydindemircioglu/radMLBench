
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
sys.path.append("./")
import radMLBench

from matplotlib.ticker import FuncFormatter


def add_percentage(row):
    total = row['Morphological'] + row['Intensity'] + row['Textural'] + row['Fractal'] + row['Other']
    return [
        f"{row['Morphological']} ({row['Morphological']/total:.1%})",
        f"{row['Intensity']} ({row['Intensity']/total:.1%})",
        f"{row['Textural']} ({row['Textural']/total:.1%})",
        f"{row['Fractal']} ({row['Fractal']/total:.1%})",
        f"{row['Other']} ({row['Other']/total:.1%})"
    ]

def tick_formatter(x, pos):
    return '{:.0f}'.format(x)

def generateNdPlot(df):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='#Instances', y='#Features',  data=df)#, legend=False)
    plt.ylabel('Number of features', fontsize=12)
    plt.xlabel('Sample size', fontsize=12)
    plt.gca().set_facecolor('white')
    plt.grid(False)
    plt.xscale('log')
    plt.yscale('log')

    x_limits = plt.xlim()
    y_limits = plt.ylim()
    plt.plot([0, 2000], [0, 2000], linestyle='dashed', color='grey')
    plt.xlim(x_limits)
    plt.ylim(y_limits)

    formatter = FuncFormatter(tick_formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xticks([50, 100, 200, 500, 1000])
    plt.yticks([100, 200, 500, 1000, 2000, 5000, 10000])

    plt.savefig('./paper/Figure_2.png', dpi=400)



if __name__ == '__main__':
    tbl = []
    for dataset in radMLBench.listDatasets():
        print (dataset)
        X = radMLBench.loadData(dataset, return_X_y = False, local_cache_dir = "./datasets")

        feats = {}
        feats["Dataset"] = dataset

        y = radMLBench.getMetaData(dataset)
        feats["#Instances"] = y["nInstances"]
        feats["#Features"] = y["nFeatures"]
        feats["Dimensionality"] = y["Dimensionality"]
        feats["ClassBalance"] = y["ClassBalance"]


        if y["Missings"] > 0:
            feats["Missing"] = f'{(y["Missings"]/(y["nInstances"]*y["nFeatures"])):.1%}'
        else:
            feats["Missing"] = "-"

        feats["Morphological"] = 0
        feats["Intensity"] = 0
        feats["Textural"] = 0
        feats["Fractal"] = 0
        feats["Other"] = 0

        pyrad = False
        for k in X.keys():
            if "original_shape_Elongation" in k:
                pyrad = True
        if pyrad == False:
            # check if we filled it out already?
            ffeattypes = os.path.join(f"./raw/{dataset}/feattypes.xlsx")
            try:
                feattypes = pd.read_excel(ffeattypes)
            except:
                print (f"No pyrad: {dataset}, please fix feature types.")
                tmpdf = [{"Feature": k, "Type": ''} for k in X.keys()]
                tmpdf = pd.DataFrame(tmpdf)
                tmpdf.to_excel(ffeattypes, index=False)

            # can check even if we just wrote that file
            for z in [("Morphological", "M"), ("Intensity", "I"), ("Textural", "T"),
                        ("Fractal", "F"), ("Other", "O")]:
                fn = z[0]
                fa = z[1]
                q = feattypes.query("Type == @fa")
                feats[fn] = len(q)
            fkeys = ['Morphological', 'Intensity', 'Textural', 'Fractal', 'Other']
            total_feats_count = sum(feats[key] for key in fkeys)
            assert (total_feats_count+2 == len(feattypes)) # +2=ID,Target
        else:
            # deng2023 and song2020 have diagnostics keys, but these are used
            # as number of voxel/volume/intensity min,max etc
            for k in sorted(X.keys()):
                if "_glcm_" in k or "_gldm_" in k or "_gldzm_" in k or "_glrlm_" in k or "_glszm_" in k or "_ngtdm_" in k:
                    feats["Textural"] += 1
                elif "_shape_" in k or "vol_" in k or "general_info_Vo" in k or "diagnostics_Mask-inter" in k:
                    feats["Morphological"] += 1
                elif "firstorder_" in k or "diagnostics_Image-inter" in k or "diagnostics_Image-original" in k or "diagnostics_Mask-origi" in k:
                    feats["Intensity"] += 1
                elif "ID" == k or "Target" == k:
                    pass
                else:
                    raise Exception (f"Unknown feature: {k}")

        tbl.append(feats)
    table2 = pd.DataFrame(tbl)
    table2['Missing'] = table2['Missing'].apply(lambda x: '<0.1%' if x == '0.0%' else x)
    table2["D"] = table2["Morphological"] + table2["Intensity"] + table2["Textural"] + table2["Fractal"] + table2["Other"]
    assert (table2["D"] == table2["#Features"]).all()
    table2[['Morphological', 'Intensity', 'Textural', 'Fractal', 'Other']] = table2.apply(add_percentage, axis=1, result_type='expand')
    table2.to_excel("./paper/Table2.xlsx", index = False)

    print ("Data stats:")
    print ("#Instances", np.min(table2["#Instances"]), np.max(table2["#Instances"]))
    print ("#Features", np.min(table2["#Features"]), np.max(table2["#Features"]))
    generateNdPlot(table2)


#
