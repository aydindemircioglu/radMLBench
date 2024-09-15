
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.path.append("./")
import radMLBench


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
                    "Modality": y["modality"]})
    table1 = pd.DataFrame(table1)
    table1.to_excel("./paper/Table1.xlsx", index = False)


#
