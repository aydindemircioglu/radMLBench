
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.path.append("./")
import radMLBench

from matplotlib.ticker import FuncFormatter


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



def inject_table(df, filename):
    with open(filename, 'r') as file:
        content = file.read()

    start_tag = "<!-- dataset table:start -->"
    end_tag = "<!-- dataset table:end -->"
    start_index = content.find(start_tag)
    end_index = content.find(end_tag)

    if start_index != -1 and end_index != -1:
        updated_content = content[:start_index + len(start_tag)] + "\n" + content[end_index:]
        markdown_table = df.to_markdown(index=False)
        updated_content = updated_content.replace(end_tag, f"\n{markdown_table}\n{end_tag}")
        with open(filename, 'w') as file:
            file.write(updated_content)
    else:
        print("Start or end tag not found in the file.")



if __name__ == '__main__':
    tableREADME = []
    for dset in radMLBench.listDatasets():
        y = radMLBench.getMetaData(dset)
        tableREADME.append({"Dataset": dset,
                    "Year": y["year"],
                    "Modality": y["modality"],
                    "#Instances": y["nInstances"],
                    "#Features": y["nFeatures"],
                    "Dimensionality": y["Dimensionality"],
                    "ClassBalance": y["ClassBalance"],
                    "DOI": y["publication_doi"]})
    tableREADME = pd.DataFrame(tableREADME)
    inject_table(tableREADME, "./README.md")

    table1 = []
    for dset in radMLBench.listDatasets():
        y = radMLBench.getMetaData(dset)
        table1.append({"Dataset": dset,
                    "Year": y["year"],
                    "Pathology": y["pathology"],
                    "Outcome": y["outcome"],
                    "Modality": y["modality"],
                    "#Instances": y["nInstances"],
                    "#Features": y["nFeatures"],
                    "Dimensionality": y["Dimensionality"],
                    "ClassBalance": y["ClassBalance"]})
    table1 = pd.DataFrame(table1)
    table1.to_excel("./paper/Table1.xlsx", index = False)

    #missings
    print ("Missings:")
    for dset in radMLBench.listDatasets():
        y = radMLBench.getMetaData(dset)
        if y["Missings"] > 0:
            print (y["dataset"], y["Missings"], 100*y["Missings"]/(y["nInstances"]*y["nFeatures"]))

    print ("Data stats:")
    print ("#Instances", np.min(tableREADME["#Instances"]), np.max(tableREADME["#Instances"]))
    print ("#Features", np.min(tableREADME["#Features"]), np.max(tableREADME["#Features"]))
    generateNdPlot(table1)


#
