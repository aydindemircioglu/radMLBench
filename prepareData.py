import numpy as np
import os
import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
import yaml #? or pyaml?


import re

from sklearn.preprocessing import StandardScaler

from glob import glob
import shutil


radiomics_uk_datasets = ["BraTS-2021",
            "C4KC-KiTS",
            "Colorectal-Liver-Metastases",
            "HCC-TACE-Seg",
            "Head-Neck-PET-CT",
            "Head-Neck-Radiomics-HN1",
            "HNSCC",
            "LGG-1p19qDeletion",
            # "LIDC-IDRI",low-dim < 0.5
            "LNDb",
            # "LUAD-CT-Survival", also contains <50 pats after preprocessing
            "Meningioma-SEG-CLASS",
            "NSCLC-Radiogenomics",
            #"NSCLC-Radiomics", same as Hosny2018B (Maastro, i believe)
            # "OPC-Radiomics", low-dim < 0.5
            "PI-CAI",
            "Prostate-MRI-US-Biopsy",
            "QIN-HEADNECK",
            # REMOVE  "Soft-tissue-Sarcoma", remove, because 51 pats, and would less than 50 if survival or grade would be used
            "UCSF-PDGM",
            "UPENN-GBM",
            "WORC-CRLM",
            "WORC-Desmoid",
            "WORC-GIST",
            "WORC-Lipo",
            "WORC-Liver",
            "WORC-Melanoma"]



def recreatePath (path, create = True):
    print ("Recreating path ", path)
    try:
        shutil.rmtree (path)
    except:
        pass

    if create == True:
        try:
            os.makedirs (path)
        except:
            pass
    print ("Done.")



def readYaml(file_path):
    try:
        with open(file_path, 'r') as stream:
            data = yaml.safe_load(stream)
            return data
    except Exception as e:
        #print (e)
        return None



def getMetadata (data_name, folder = "./raw"):
    # first try direct
    yfile = os.path.join(folder, data_name, "metadata.yaml")
    y = readYaml(yfile)
    if y is None:
        # try radiomics.uk as well
        yfile = os.path.join(folder, "./radiomics.uk", f"{data_name}_metadata.yaml")
        y = readYaml(yfile)
    return y



def fixDataName (sub_data_name):
    if sub_data_name == "UPENN-GBM":
        sub_data_name = "UPENN-GBM_enhancing_tumor"
    if sub_data_name == "UCSF-PDGM":
        sub_data_name = "UCSF-PDGM_enhancing_tumor"
    if sub_data_name == "QIN-HEADNECK":
        sub_data_name = "QIN-HEADNECK_CT"
    if sub_data_name == "Prostate-MRI-US-Biopsy":
        sub_data_name = "Prostate-MRI-US-Biopsy_prostate"
    return sub_data_name



def fixTargets (data, data_name):
    if data_name == "LIDC-IDRI":
        newTargets = []
        # do not care about the rest
        for k in data.keys():
            nk = ''
            if "Nodule 1" in k:
                nk = nk + "Nodule_1"
            if "Nodule 2" in k:
                nk = nk + "Nodule_2"
            if 'Method' in k:
                nk = nk + "_Method"
            if nk == '':
                newTargets.append(k)
            else:
                newTargets.append(nk)
        data.columns = newTargets
    return data


def fixID (data, data_name):
    try:
        pIDs = ["patient_id", "Patient-ID", "TCIA_ID", "Patient #", "id",
                "TCIA Radiomics dummy ID of To_Submit_Final", "Patient ID",
                "Filename", "TCIA Patient ID", "LNDbID", "Patient", "case_ID",
                "Trial PatientID", "PatientID", "ID", "Subject", "subject_ID"]
        data = data.rename(columns={p: 'patient_ID' for p in pIDs})

        # now special cases, why not, let there be special case, thats a song by tocotronic, (c) 1966. i am sure.
        if data_name == "LNDb":
            # only for targets
            if len(str(data["patient_ID"].values[0])) < 3:
                data["patient_ID"] = data["patient_ID"].apply(lambda x: f'LNDb-{x:04d}')

        if data_name == "UCSF-PDGM":
            # thank you chatgpt.
            def normalize_patient_ID(patient_ID):
                pattern = r'UCSF-PDGM-(\d{1,4})'  # Define a regex pattern to match the patient ID format
                match = re.match(pattern, patient_ID)
                if match:
                    number = match.group(1).zfill(4)  # Extract the number and pad it with leading zeros
                    return f"UCSF-PDGM-{number}"
                else:
                    return patient_ID  # Return unchanged if the format doesn't match

            data["patient_ID"] = data["patient_ID"].apply(normalize_patient_ID)


        if 'patient_ID' not in data.columns:
            print (data.columns)
            raise ValueError("Column 'patient_ID' does not exist after renaming.")
    except ValueError as e:
        print(e)
    return data


def getTarget (data_name):
    # case-by-case........ yes, thank you.
    targetList = {"BraTS-2021": "MGMT_value",
        "C4KC-KiTS" : "tumor_histologic_subtype",
        "Colorectal-Liver-Metastases" : "overall_survival_months",
        "HCC-TACE-Seg" : "Lymphnodes",
        "Head-Neck-PET-CT" : "N-stage",
        "Head-Neck-Radiomics-HN1" : "clin_n",
        "HNSCC" : "N-category",
        "LGG-1p19qDeletion" : "1p/19q",
        # "LIDC-IDRI" : "Nodule_1",
        "LNDb" : "Fleischner",
        # "LUAD-CT-Survival" : "survival_label",
        "Meningioma-SEG-CLASS" : "Pathologic grade",
        "NSCLC-Radiogenomics" : "EGFR mutation status",
        # "NSCLC-Radiomics" : "Survival.time",
        # "OPC-Radiomics" : "N",
        "PI-CAI": "lesion_GS",
        "Prostate-MRI-US-Biopsy": "max_gleason",
        "QIN-HEADNECK": "N Stage",
        # "Soft-tissue-Sarcoma" : "Time – diagnosis to last follow-up (days)",
        "UCSF-PDGM": "IDH",
        "UPENN-GBM": "MGMT",
        "WORC-CRLM" : "Diagnosis_binary",
        "WORC-Desmoid" : "Diagnosis_binary",
        "WORC-GIST" : "Diagnosis_binary",
        "WORC-Lipo" : "Diagnosis_binary",
        "WORC-Liver" : "Diagnosis_binary",
        "WORC-Melanoma" : "Diagnosis_binary"}
    return targetList[data_name]



class RadTabularDataLoader():
    """ Loading works different for radiomics_uk, because of historical reasons.
        I wanted to use an XAI package so used their class structure to load the
        data, and copied it instead of rewriting """
    def __init__(self, data_name, folder = "./radiomics_uk/data", scale='minmax'):
        self.data_name = data_name
        try:
            sub_data_name = fixDataName(data_name)

            self.dataset = pd.read_csv(f"./{folder}/{sub_data_name}_features").reset_index(drop = True)
            self.targets = pd.read_csv(f"./{folder}/{data_name}_labels").reset_index(drop = True)

            self.dataset = fixID (self.dataset, data_name)
            self.targets = fixID (self.targets, data_name)
            self.targets = fixTargets (self.targets, data_name)

            if data_name == "PI-CAI":
                # special case again.........
                self.dataset["study_id"] = self.dataset["study_ID"]
                merged_df = pd.merge(self.dataset, self.targets, on=['patient_ID', 'study_id'], how='left')
            elif data_name == "Prostate-MRI-US-Biopsy":
                # special case again.........
                merged_df = pd.merge(self.dataset, self.targets, on=['patient_ID', 'img_series_instance_UID'], how='left')
            else:
                merged_df = pd.merge(self.dataset, self.targets, on='patient_ID', how='left')

            assert (len(merged_df) == len(self.dataset))
        except Exception as e:
            print(e)
            raise RuntimeError(f"Dataset not found. Possibly wrong dataset name? Tried to load ./data/{data_name}_features")
        #print (merged_df.iloc[0:50])

        self.target_name = getTarget(data_name)
        self.dataset = merged_df


        sequences = None
        if data_name == "BraTS-2021":
            self.dataset = self.dataset.query("ROI == 'enhancing_tumor'").copy().reset_index()

        if data_name == "C4KC-KiTS":
            # before we can do that, we need to remove Tumor_2/3/4/5..
            self.dataset["sequence"] = self.dataset["phase"]
            self.dataset[self.target_name] = self.dataset[self.target_name] == "clear_cell_rcc"

        if data_name == "Colorectal-Liver-Metastases":
            self.dataset = self.dataset.query (f"`vital_status` == 0").copy().reset_index(drop = True)
            self.dataset["sequence"] = self.dataset["ROI"]
            sequences = set(self.dataset["sequence"])
            sequences = [k for k in sequences if "Tumor" not in k or k == "Tumor_1"]
            self.dataset[self.target_name] = self.dataset[self.target_name] < 12*10

        if data_name == "HCC-TACE-Seg":
            self.dataset[self.target_name] = self.dataset[self.target_name]
            self.dataset["sequence"] = self.dataset["ROI"]
            sequences = set(self.dataset["sequence"])

        if data_name == "Head-Neck-PET-CT":
            self.dataset = self.dataset.query (f"`{self.target_name}` == `{self.target_name}`").copy().reset_index(drop = True)
            self.dataset["sequence"] = self.dataset["modality"]
            sequences = set(self.dataset["modality"])
            self.dataset[self.target_name] = (self.dataset[self.target_name] != "N0") & (self.dataset[self.target_name] != "N1").astype(int)

        if data_name == "Head-Neck-Radiomics-HN1":
            self.dataset = self.dataset.query (f"`{self.target_name}` == `{self.target_name}`").copy().reset_index(drop = True)
            self.dataset["sequence"] = self.dataset["ROI"]
            sequences = ["GTV-1"]
            self.dataset[self.target_name] = self.dataset[self.target_name] > 1

        if data_name == "HNSCC":
            self.dataset = self.dataset.query (f"`{self.target_name}` != 'Unknown'").copy().reset_index(drop = True)
            self.dataset[self.target_name] = self.dataset[self.target_name] > 1
            self.dataset["sequence"] = self.dataset["ROI"]
            sequences = ["GTV"]

        if data_name == "LGG-1p19qDeletion":
            self.dataset[self.target_name] = self.dataset[self.target_name] == "d/d"
            sequences = set(self.dataset["sequence"])

        if data_name == "LIDC-IDRI":
            # rename target
            self.dataset = self.dataset.query (f"`{self.target_name}` != 0").copy().reset_index(drop = True)
            self.dataset = self.dataset.query("`seg_ID`.str.contains(r'Annotation_Nodule')", engine='python').copy().reset_index(drop=True)
            # thats not enough, we drop duplicates then
            self.dataset.drop_duplicates(subset="patient_ID", inplace=True)
            self.dataset[self.target_name] = self.dataset[self.target_name] > 1
            self.dataset["sequence"] = self.dataset["ROI_ID"]
            sequences = ["Nodule_1"]

        if data_name == "LNDb":
            # rename target
            self.dataset = self.dataset.query("`seg_ID`.str.contains(r'finding1$')", engine='python').copy().reset_index(drop=True)
            self.dataset[self.target_name] = self.dataset[self.target_name] > 1
            self.dataset["sequence"] = self.dataset["reader_ID"]
            sequences = [1]

        if data_name == "LUAD-CT-Survival":
            # rename target
            self.dataset[self.target_name] = self.dataset[self.target_name] == "Long"
            self.dataset["sequence"] = "CT"

        if data_name == "Meningioma-SEG-CLASS":
            # rename target
            self.dataset[self.target_name] = self.dataset[self.target_name] == "II"
            self.dataset["sequence"] = ["FLAIR" if "flair" in k.lower() else "T1" for k in self.dataset["sequence"]]

        if data_name == "NSCLC-Radiogenomics":
            self.dataset = self.dataset.query (f"`{self.target_name}` != 'Not collected'").copy().reset_index(drop = True)
            self.dataset[self.target_name] = self.dataset[self.target_name] == "Mutant"
            self.dataset["sequence"] = "CT"

        if data_name == "NSCLC-Radiomics":
            # ignore those with censored data
            self.dataset = self.dataset.query (f"`deadstatus.event` == 1").copy().reset_index(drop = True)
            self.dataset[self.target_name] = self.dataset[self.target_name] > 365*3 # 3 year
            self.dataset["sequence"] = "primary tumor"


        if data_name == "OPC-Radiomics":
            self.dataset = self.dataset.query (f"ROI == 'GTV'").copy().reset_index(drop = True)
            self.dataset[self.target_name] = (self.dataset[self.target_name] != "N0") & (self.dataset[self.target_name] != "N1").astype(int)
            self.dataset["sequence"] = "GTV"

        if data_name == "PI-CAI":
            self.dataset = self.dataset.query (f"annotator == 'AI'").copy().reset_index(drop = True)
            self.dataset = self.dataset.query (f"ROI == 'prostate'").copy().reset_index(drop = True)
            self.dataset = self.dataset.query (f"{self.target_name} == {self.target_name}").copy().reset_index(drop = True)
            # remove duplicates...
            studyIDs = self.dataset[["patient_ID", "study_ID"]].copy().drop_duplicates(["patient_ID"])["study_ID"]
            self.dataset = self.dataset.query('study_ID in @studyIDs').copy()

            newValues = []
            for z in self.dataset[self.target_name].values:
                mv = -1
                # here we only use 0+0 vs rest, so ifs are actually not needed.
                for k in z.split(","):
                    # will always be of the form A+B, and we do not care about what exactly
                    if k == "N/A":
                        continue
                    mv = np.max([mv, int(k[0])+int(k[2])])
                if mv == -1:
                    raise Exception ("!!!!")
                if mv > 0:
                    mv = 1
                newValues.append(mv)
            self.dataset[self.target_name] = newValues


        if data_name == "Prostate-MRI-US-Biopsy":
            self.dataset[self.target_name] = self.dataset[self.target_name] > 1.0
            self.dataset["sequence"] = "lesion"
            # can remove duplicates because only one sequence per patient
            self.dataset.drop_duplicates(subset="patient_ID", inplace=True)

        if data_name == "QIN-HEADNECK":
            self.dataset = self.dataset.query (f"timepoint == 1").copy().reset_index(drop = True)
            self.dataset = self.dataset.query (f"roi == 'neoplasm_primary_1' ").copy().reset_index(drop = True)
            self.dataset = self.dataset.query (f"reader == 'user1' ").copy().reset_index(drop = True)
            self.dataset = self.dataset.query (f"trial_num == 1 ").copy().reset_index(drop = True)
            self.dataset[self.target_name] = (self.dataset[self.target_name] != "0") & (self.dataset[self.target_name] != "1").astype(int)
            self.dataset["sequence"] = self.dataset["segmentation_method"]

        if data_name == "UCSF-PDGM":
            self.dataset = self.dataset.query (f"{self.target_name} == {self.target_name}").copy().reset_index(drop = True)
            self.dataset[self.target_name] = self.dataset[self.target_name] == "wildtype"
            #print (self.dataset[self.target_name].values)
            sequences = ["T1", "T1c", "T2", "DWI", "SWI", "FLAIR", "ADC"]
            # in a few cases, at least UCSF-PDGM, we have multiple same IDs,
            # and we can only distinguish them by their unique_ID
            # so well...
            newIDs = self.dataset.apply(lambda row: f"{row['patient_ID']}_{row['sequence']}", axis=1).tolist()
            self.dataset = self.dataset.query('unique_ID in @newIDs').copy()

        if data_name == "UPENN-GBM":
            tmpvar = self.target_name
            self.dataset = self.dataset.query (f"{self.target_name} != 'Not Available'").copy().reset_index(drop = True)
            self.dataset = self.dataset.query (f"{self.target_name} != 'Indeterminate'").copy().reset_index(drop = True)
            self.dataset[self.target_name] = self.dataset[self.target_name] == "Methylated"

        if "WORC" in data_name:
            # special case.... got used to it now
            self.dataset = self.dataset.query (f"`{self.target_name}` == `{self.target_name}`").copy().reset_index(drop = True)
            self.dataset = self.dataset.query(f"`{self.target_name}` == 0 or `{self.target_name}` == 1").copy().reset_index(drop=True)
            seq = "CT"
            if "Desmoid" in data_name or "Lipo" in data_name or "Liver" in data_name:
                seq = "MR"
            self.dataset["sequence"] = seq

        if sequences is None:
            sequences = set(self.dataset['sequence'])

        self.dataset["Target"] = self.dataset[self.target_name].astype(np.uint32)

        # keys without XX_XX_XX are not radiomic
        metaKeys = [k for k in self.dataset.keys() if k.split("_")[0].split("-")[0] not in ["original", "wavelet", "log"]]

        # now the rubbish part, join the sequences
        new_dataset = []
        for ID in self.dataset['patient_ID'].unique():
            tgts = []
            patient_sequences = set(self.dataset[self.dataset['patient_ID'] == ID]['sequence'])
            has_all_sequences = np.sum([1 if k in patient_sequences else 0 for k in sequences]) == len(sequences)
            if has_all_sequences == True:
                pat_data = []
                for seq in sequences:
                    seq_data = self.dataset[(self.dataset['patient_ID'] == ID) & (self.dataset['sequence'] == seq)]
                    if len(seq_data) > 1:
                        print (seq_data)
                        exit(-1)
                    tgt = seq_data["Target"]
                    seq_data = seq_data.drop(metaKeys, axis = 1).copy()
                    seq_data.columns = [f"{seq}_{col}" for col in seq_data.columns ]
                    # seq_data["ID"] = id
                    pat_data.append(seq_data.reset_index(drop = True))
                    tgts.append(tgt.values[0])
                    # print (pat_data)
                # dump (pat_data, "delme.dump")
                # pat_data = load("delme.dump")
                assert (len(set(tgts)) == 1)
                tmp = pd.concat(pat_data, axis = 1)
                tmp["ID"] = ID
                tmp["Target"] = list(set(tgts))[0]
                if len(tmp) != 1:
                    print (tmp)
                    exit(-1)
                assert (len(tmp) == 1)
                new_dataset.append( tmp)
        self.X = pd.concat(new_dataset, ignore_index=True).copy()
        #self.X["Target"] = self.dataset["Target"]
        self.X.target_name = "Target"

        # this is soo rubbish.
        self.X = self.X.applymap(lambda x: complex(x).real if isinstance(x, str) and 'j' in x else x)
        self.feature_names = self.X.columns.to_list()
        self.feature_types = ['c']*len(self.feature_names)
        self.feature_metadata = self.feature_types

        self.data = self.X.copy()


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx = idx.tolist() if isinstance(idx, torch.Tensor) else idx
        return (self.data[idx], self.targets.values[idx])

    def get_number_of_features(self):
        return self.data.shape[1]

    def get_number_of_instances(self):
        return self.data.shape[0]



def checkData (df):
    # ensure we have patient and target
    try:
        l = len(df.ID.values)
    except:
        raise Exception ("ID is missing in the dataset!")
    assert (len(df) == len(set(df.ID))), "Not all IDs are unique!"



folder = './raw/Ahn2021'
def Ahn2021 (folder = None):
    inputFile = os.path.join(folder, "data/EJR_N114_20180930_raw_data.csv")
    df = pd.read_csv(inputFile)
    df["Target"] = df['mgmt']
    df["ID"] = df["Patient"]
    df = df.drop(['resection', "Patient", "mgmt", 'sex', 'age', 'time', 'status'], axis = 1)
    df = df.reset_index(drop = True)
    return df.copy()



folder = './raw/Arita2018'
def Arita2018 (folder = None):
    inputFile = "41598_2018_30273_MOESM3_ESM.csv"
    dataA = pd.read_csv(os.path.join(folder, "data", inputFile), encoding = "ISO-8859-1")

    inputFile = "41598_2018_30273_MOESM4_ESM.csv"
    dataB = pd.read_csv(os.path.join(folder, "data", inputFile), encoding = "ISO-8859-1")
    data = pd.concat([dataA,dataB])
    data["Target"] = data["IDH.1"]
    data["ID"] = data["Pt. ID"]
    data = data[[z for z in data.keys()[33:]]]
    data = data[data.isnull().sum(axis=1) < 22]
    data = data.drop(["Image_ID"], axis  = 1)
    data = data.reset_index(drop = True)
    return data.copy()



folder = './raw/Brancato2023'
def Brancato2023 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "FEATURES_Batch_Effect.xlsx"
    dataShape = pd.read_excel(os.path.join(dataDir, inputFile), sheet_name = 3)
    dataT2 = pd.read_excel(os.path.join(dataDir, inputFile), sheet_name = 4)
    dataADC = pd.read_excel(os.path.join(dataDir, inputFile), sheet_name = 5)
    dataT2 = dataT2.drop(["PATIENT", "BIOPSY"], axis = 1)
    dataT2.columns = [f"T2_{col}" for col in dataT2.columns]
    dataADC = dataADC.drop(["PATIENT", "BIOPSY"], axis = 1)
    dataADC.columns = [f"ADC_{col}" for col in dataADC.columns]
    data = pd.concat([dataShape,dataT2,dataADC], axis = 1)
    data["ID"] = data["PATIENT"]
    data["Target"] = 1 - (data["BIOPSY"] == "no-PCa")
    data = data.drop(["PATIENT", "BIOPSY"], axis  = 1)
    return data.copy()



folder = './raw/Carvalho2018'
def Carvalho2018 (folder = None):
    inputFile = "Radiomics.PET.features.csv"
    data = pd.read_csv(os.path.join(folder, "data", inputFile))
    # all patients that are lost to followup were at least followed for two
    # years. that means if we just binarize the followup time using two years
    # we get those who died or did not die within 2 years as binary label
    data["Target"] = (data["Survival"] < 2.0)*1
    data = data.drop(["Survival", "Status"], axis = 1)
    data = data.reset_index(drop = True).reset_index(drop = False)
    data["ID"] = data['index']
    data = data.drop(['index'], axis = 1)
    return data.copy()



folder = './raw/Deng2023'
def Deng2023 (folder = None):
    inputFile = "testset_t1c.xlsx"
    dataA = pd.read_excel(os.path.join(folder, "data", inputFile))
    inputFile = "trainset_t1c.xlsx"
    dataB = pd.read_excel(os.path.join(folder, "data", inputFile))
    inputFile = 'yueyang_t1c_N4_re.xlsx'
    dataC = pd.read_excel(os.path.join(folder, "data", inputFile))

    data = pd.concat([dataA,dataB,dataC])
    data["Target"] = data["pathology"]
    data = data.reset_index(drop = True).reset_index(drop = False)
    data["ID"] = data["index"]
    #data = data.drop([k for k in data.keys() if "diagn" in k], axis = 1).copy()
    data = data.drop(['index', 'set', 'pathology'], axis = 1)
    return data.copy()



folder = './raw/Granata2021'
def Granata2021 (folder = None):
    inputFile = os.path.join(folder, "data/database immuno.xlsx")
    data = pd.read_excel(inputFile)
    osurv = data["OS"]
    data["Target"] = (osurv > 32).astype(np.uint32)
    data = data.reset_index(drop = True).reset_index(drop = False)
    data["ID"] = data["index"]
    data = data.drop(["OS", "PFS", "Experimental group versus control group\n", 'index'], axis = 1)
    return data.copy()



folder = './raw/Granata2024'
def Granata2024 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "Database on arterial phase.xlsx"
    data = pd.read_excel(os.path.join(dataDir, inputFile))
    data = data.drop_duplicates()

    data["Target"] = data["BUDDING"]
    data = data.reset_index(drop = True).reset_index(drop = False)
    data["ID"] = data['index']
    data = data.drop(['index', 'BUDDING'], axis = 1)
    return data.copy()




folder = './raw/Hosny2018A'
def Hosny2018A (folder = None):
    dataDir = os.path.join(folder, "data/")
    # take only HarvardRT
    data = pd.read_csv(os.path.join(dataDir, "HarvardRT.csv"))
    data = data.drop([z for z in data.keys() if "general_" in z], axis = 1)
    data["Target"] = data['surv2yr']
    # logit_0/logit_1 are possibly output of the CNN network
    dataID = data['id']
    data = data.drop(['id', 'surv2yr', 'logit_0', 'logit_1'], axis = 1)

    # fix power transform overflow bugs by removing the features
    L = list(data.keys())[0:825]+list(data.keys())[845:1004]
    data = data[L+["Target"]].copy()
    data["ID"] = dataID
    data = data.reset_index(drop = True)
    return data.copy()




folder = './raw/Hosny2018B'
def Hosny2018B (folder = None):
    dataDir = os.path.join(folder, "data/")

    data = pd.read_csv(os.path.join(dataDir, "Maastro.csv"))
    data = data.drop([z for z in data.keys() if "general_" in z], axis = 1)
    data["Target"] = data['surv2yr']
    # logit_0/logit_1 are possibly output of the CNN network
    dataID = data['id']
    data = data.drop(['id', 'surv2yr', 'logit_0', 'logit_1'], axis = 1)

    # fix power transform overflow bugs by removing the features
    L = list(data.keys())[0:825]+list(data.keys())[845:1004]
    data = data[L+["Target"]].copy()
    data["ID"] = dataID
    data = data.reset_index(drop = True)
    data.drop_duplicates(subset='ID', keep='last', inplace=True)
    return data.copy()




folder = './raw/Hosny2018C'
def Hosny2018C (folder = None):
    dataDir = os.path.join(folder, "data/")
    # take only HarvardRT
    data = pd.read_csv(os.path.join(dataDir, "Moffitt.csv"))
    data = data.drop([z for z in data.keys() if "general_" in z], axis = 1)
    data["Target"] = data['surv2yr']
    # logit_0/logit_1 are possibly output of the CNN network
    dataID = data['id']
    data = data.drop(['id', 'surv2yr', 'logit_0', 'logit_1'], axis = 1)

    # fix power transform overflow bugs by removing the features
    L = list(data.keys())[0:825]+list(data.keys())[845:1004]
    data = data[L+["Target"]].copy()
    data["ID"] = dataID
    data = data.reset_index(drop = True)
    return data.copy()



folder = './raw/Hunter2023'
def Hunter2023 (folder = None):
    inputFileTrain = os.path.join(folder, "data/Features_Train.csv")
    dftrain = pd.read_csv(inputFileTrain)

    inputFileTest = os.path.join(folder, "data/Features_Test.csv")
    dftest = pd.read_csv(inputFileTest)

    df = pd.concat([dftrain, dftest], axis = 0)
    df = df.reset_index(drop = True).copy()
    df = df.dropna(subset=['Lesion_N.voxels']).copy()
    df = df.dropna(axis=1).copy()
    df[['Patient_ID', 'Lesion_Number']] = df['ID'].str.split('_', expand=True)
    idx_max_lesion = df.groupby('Patient_ID')['Lesion_N.voxels'].idxmax()
    result_df = df.loc[idx_max_lesion]
    result_df = result_df.copy()

    # Drop unnecessary columns
    result_df["Target"] = result_df["Outcome"]
    result_df = result_df.drop(['Lesion_Number', 'Patient_ID', 'Outcome'], axis=1).copy()
    return result_df.copy()




folder = './raw/ISPY1'
#             infoTbl.append({"Patient": ID, "Diagnosis": int(row["HR Pos"])})
def ISPY1 (folder = None):
    inputFile = os.path.join(folder, "./data/ISPY_DataPaper_features.xlsx")
    data = pd.read_excel(inputFile)
    inputFile = os.path.join(folder, "./data/pInfo_ISPY1.csv")
    clin = pd.read_csv(inputFile)
    data["SubjectID"] = data["SubjectID"].astype(str)
    clin["SubjectID"] = [k.replace("ISPY1-", "") for k in clin["Patient"].values]
    data = data.merge(clin, on='SubjectID', how='inner')
    data["ID"] = data["Patient"]
    data["Target"] = data["Diagnosis"]
    data = data.drop(['Diagnosis', 'SubjectID', 'Patient'], axis=1).copy()
    data = data.reset_index(drop = True)
    return data.copy()




folder = './raw/ISPY2'
def ISPY2 (folder = None):
    inputFile = os.path.join(folder, "./data/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx")
    data = pd.read_excel(inputFile)

    inputFile = os.path.join(folder, "./data/train.xlsx")
    data = pd.read_excel(inputFile)

    data.shape


    inputFile = os.path.join(folder, "./data/pInfo_ISPY1.csv")
    clin = pd.read_csv(inputFile)
    data["SubjectID"] = data["SubjectID"].astype(str)
    clin["SubjectID"] = [k.replace("ISPY1-", "") for k in clin["Patient"].values]
    data = data.merge(clin, on='SubjectID', how='inner')
    data["ID"] = data["Patient"]
    data["Target"] = data["Diagnosis"]
    data = data.drop(['Diagnosis', 'SubjectID', 'Patient'], axis=1).copy()
    data = data.reset_index(drop = True)
    return data.copy()



folder = './raw/Keek2020'
def Keek2020 (folder = None):
    inputFile = "Clinical_DESIGN.csv"
    clDESIGNdata = pd.read_csv(os.path.join(folder, "data", inputFile), sep=";")

    df = clDESIGNdata.copy()
    # remove those pats who did not die and have FU time less than 3 years
    df = clDESIGNdata[(clDESIGNdata["StatusDeath"].values == 1) | (clDESIGNdata["TimeToDeathOrLastFU"].values > 3*365)]
    target = df["TimeToDeathOrLastFU"] < 3*365
    target = np.asarray(target, dtype = np.uint8)
    id =     clDESIGNdata.StudySubjectID

    inputFile = "Radiomics_DESIGN.csv"
    rDESIGNdata = pd.read_csv(os.path.join(folder, "data", inputFile), sep=";")
    rDESIGNdata = rDESIGNdata.drop([z for z in rDESIGNdata.keys() if "General_" in z], axis = 1)
    rDESIGNdata = rDESIGNdata.loc[df.index]
    rDESIGNdata = rDESIGNdata.reset_index(drop = True)
    rDESIGNdata["Target"] = target

    # convert strings to float
    rDESIGNdata = rDESIGNdata.applymap(lambda x: float(str(x).replace(",", ".")))
    rDESIGNdata["Target"] = target
    rDESIGNdata["ID"] = id

    rDESIGNdata = rDESIGNdata.reset_index(drop = True)
    return rDESIGNdata.copy()



folder = './raw/Li2020'
def Li2020 (folder = None):
    inputFile = "pone.0227703.s014.csv"
    data = pd.read_csv(os.path.join(folder, "data", inputFile))
    data["Target"] = data["Label"]
    data = data.drop(["Label"], axis = 1)
    data = data.reset_index(drop = True).reset_index(drop = False)
    data["ID"] = data["index"]
    data = data.drop(['index'], axis = 1)
    return data.copy()



folder = './raw/Lu2019'
def Lu2019 (folder = None):
    inputFile = "./x_complete_scale.csv"
    data = pd.read_csv(os.path.join(folder, "data", inputFile), sep = ";")
    inputFile = "./x_complete.csv"
    target = pd.read_csv(os.path.join(folder, "data", inputFile), sep = ";")
    data["PFS"] = target["PFS.event"]
    data["PFS_time"] = target["Progression.free.survival..days."]
    data = data[data["PFS_time"].notna()]

    data = data.query("PFS == 1").copy()

    data["Target"] = (data["PFS_time"] < 365*2) & (data["PFS"] == 1)
    data["Target"] = (1.0*data["Target"]).astype(np.uint32)
    data["ID"] = target["Patient_Idold"]
    data["Code"] = target["Bilateralcode"]
    data = data.query("Code > 1").copy()
    data = data.drop(["PFS", "PFS_time", "Code"], axis = 1)
    data = data.reset_index(drop = True)
    return data.copy()




folder = './raw/Naseri2023'
def Naseri2023 (folder = None):
    inputFile = "featurespace_metadata.json"
    data = pd.read_json(os.path.join(folder, "data", inputFile))
    # CY15 seems best without ensemble? https://www.nature.com/articles/s41598-022-13379-8#Sec14
    X = pd.DataFrame(data.loc["CY15"])
    expanded_df = pd.json_normalize(data.loc['CY15'])
    expanded_df["Patient"] = X.index.astype(str)
    # add targets to values
    targets = pd.DataFrame(data.loc['label'])
    targets = targets.reset_index(drop = False)
    targets["Patient"] = targets["index"]
    merged_df = expanded_df.merge(targets, on='Patient')
    merged_df["Target"] = (merged_df["label"] == 'metastatic').astype(np.uint32)
    merged_df["ID"] = merged_df["index"]
    merged_df = merged_df.drop(["index", 'Patient', 'label'], axis = 1)
    return merged_df.copy()



folder = './raw/Park2020'
def Park2020 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "pone.0227315.s003.xlsx"
    data = pd.read_excel(os.path.join(dataDir, inputFile), engine='openpyxl')
    target = data["pathological lateral LNM 0=no, 1=yes"]
    id = data["Patient No."]
    data = data.drop(["Patient No.", "pathological lateral LNM 0=no, 1=yes",
        "Sex 0=female, 1=male", "pathological central LNM 0=no, 1=yes"], axis = 1)
    data["Target"] = target
    data["ID"] = id
    data = data.reset_index(drop = True)
    return data.copy()



folder = './raw/Petrillo2023'
def Petrillo2023 (folder = None):
    inputFile = os.path.join(folder, "data/Database_Features.xlsx")
    df = pd.read_excel(inputFile) # only take first sheet, because ID of others are..starnge
    df.drop_duplicates(subset='ID', keep='last', inplace=True)
    target = df["HR+"]
    id = df["ID"]
    df = df.drop([k for k in df.keys() if "_" not in k], axis = 1)
    df["ID"] = id
    df["Target"] = target
    return df.copy()



folder = './raw/Ramella2018'
def Ramella2018 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "pone.0207455.s001.arff"

    data = arff.loadarff(os.path.join(dataDir, inputFile))
    data = pd.DataFrame(data[0])
    data["Target"] = np.asarray(data['adaptive'], dtype = np.uint8)
    data = data.drop(['sesso', 'fumo', 'anni', 'T', 'N', "stadio", "istologia", "mutazione_EGFR", "mutazione_ALK", "adaptive"], axis = 1)
    data = data.reset_index(drop = True).reset_index(drop = False)
    data["ID"] = data["index"]
    data = data.drop(['index'], axis = 1)
    return data.copy()



folder = './raw/Sasaki2019'
def Sasaki2019 (folder = None):
    # take mgmt as target, but we have more features,
    # paper only has 489 and we only take complete cases
    dataDir = os.path.join(folder, "data")
    inputFile = "41598_2019_50849_MOESM3_ESM.xlsx"
    data = pd.read_excel(os.path.join(dataDir, inputFile),header=1, engine='openpyxl')
    data["Target"] = data["MGMT_1Met0Unmet"]
    id = data["ID"]
    data = data.drop(data.keys()[0:26], axis = 1)
    data["ID"] = id
    # complete cases only
    data = data.dropna()
    data = data.reset_index(drop = True)
    return data.copy()



folder = './raw/Song2020'
def Song2020 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "numeric_feature.csv"
    data = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
    data["Target"] = np.asarray(data["label"] > 0.5, dtype = np.uint8)
    data["ID"] = data["Unnamed: 0"]
    data = data.drop(["Unnamed: 0", "label"], axis = 1)
    data = data.reset_index(drop = True)
    return data.copy()



folder = './raw/OcanaTienda2023'
def OcanaTienda2023 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "OpenBTAI_RADIOMICS.xlsx"
    data = pd.read_excel(os.path.join(dataDir, inputFile))
    inputFile = "OpenBTAI_METS_ClinicalData_Nov2023.xlsx"
    clin = pd.read_excel(os.path.join(dataDir, inputFile))
    columns = clin.iloc[12:14].astype(str).apply('_'.join)
    detoxed_list = [s.encode('ascii', 'ignore').decode('ascii').replace(' ', '') for s in columns.values]
    clin.columns = detoxed_list
    clin = clin.query("`id_nan` == 1").copy()
    clin["Time"] = clin["Time_nan"].iloc[:, 1]
    clin["ID"] = clin["PatientID_nan"]
    clin = clin[["ID", "Time"]]

    tmp = data.query("Segment == 'Contrast-enhancing' & Lesion == 1").copy()
    tmp.sort_values(by=['Patient', 'Timepoint'], inplace=True)
    tmp.drop_duplicates(subset='Patient', keep='first', inplace=True)
    tmp["ID"] = tmp["Patient"].astype(str).str.zfill(6)
    tmp = tmp.merge(clin, on = "ID")
    tmp["Target"] = (tmp.Time > 2*365).astype(np.uint32)
    tmp = tmp.drop([k for k in tmp.keys() if "diagn" in k], axis = 1).copy()
    tmp = tmp.drop(["Patient", "Time", "Timepoint", "Label", "Lesion", "Segment", "Image", "Mask"], axis = 1).copy()
    data = tmp.reset_index(drop = True)
    return data.copy()




folder = './raw/Toivonen2019'
def Toivonen2019 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "lesion_radiomics.csv"
    data = pd.read_csv(os.path.join(dataDir, inputFile))
    data["Target"] = np.asarray(data["gleason_group"] > 0.0, dtype = np.uint8)
    id = data['id']
    data = data.drop(["gleason_group", "id"], axis = 1)
    data["ID"] = id
    # have multiple lesions? per ID, take first one
    data.drop_duplicates(subset='ID', keep='first', inplace=True)
    data = data.reset_index(drop = True)
    return data.copy()



folder = './raw/Veeraraghavan2020'
def Veeraraghavan2020 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "CERRFeatures_FINAL.csv"
    feats = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
    feats = feats.drop(["Exclude", "Histology", "FIGO", "Stage", "MolecularSubtype", "Age", "TMB", "CT_Make"], axis = 1)

    inputFile = "clinData_Nov18_2019.csv"
    targets = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
    targets = targets[["TMB", "Exclude"]]
    data = pd.concat([feats,targets], axis = 1)

    data = data[data["Exclude"] == "No"]
    data["Target"] = 1*(targets["TMB"] > 15.5)
    id = data["PID"]
    data = data.drop(["TMB", "Exclude", "PID"], axis = 1)
    data["ID"] = id
    data = data.reset_index(drop = True)
    return data.copy()



folder = './raw/Yan2023'
def Yan2023 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "12880_2023_1085_MOESM2_ESM.csv"
    gpA = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")

    dataDir = os.path.join(folder, "data")
    inputFile = "12880_2023_1085_MOESM3_ESM.csv"
    gpB = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
    data = pd.concat([gpA, gpB], axis = 0)
    id = data["id"]
    target = data["group"]
    data = data.drop(["id", "Grade", "group"], axis  = 1)
    data["ID"] = id
    data["Target"] = target
    return data.copy()




folder = './raw/Huang2023'
def Huang2023 (folder = None):
    dataDir = os.path.join(folder, "data")
    # A=arterial, V=venous, U=unenhanced
    inputFile = "U for test-new.csv"
    gpTe = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
    inputFile = "U for train.csv"
    gpTr = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
    data = pd.concat([gpTr, gpTe], axis = 0)
    data["Target"] = data['ID']
    data = data.reset_index(drop = True).reset_index(drop = False)
    data["ID"] = data['index']
    data = data.drop(['index'], axis = 1)
    return data.copy()




folder = './raw/Wang2024'
def Wang2024 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "metadata.csv"
    meta = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
    meta = meta.query("EGFRAlt == EGFRAlt").copy()

    # we first let this run, after that we knew which files to use
    # we then gzipped them and removed the rest
    # then actually the lower check is not really necessary,
    # but to be reproducible we keep it

    # some files are somehow missing, ensure we have the data
    meta['exists'] = 0
    for index, row in meta.iterrows():
        filename = os.path.join(dataDir, f"{row['Patient']}_{row['Z']}_Data.csv.gz")
        if os.path.exists(filename):
            meta.at[index, 'exists'] = 1
    meta = meta.query("exists == 1").copy()

    # randomly drop duplicates
    data = meta.drop_duplicates(subset = "Patient").copy()
    def compute_mean_features(row):
        # first read data
        filename = os.path.join(dataDir, f"{row['Patient']}_{row['Z']}_Data.csv.gz")
        subdf = pd.read_csv(filename, compression = "gzip")
        # print (f"cp {filename} ./wangnew")
        # next we simply ignore the biopsy and compute mean features
        fv = subdf.iloc[:,7:].mean(skipna=True)
        for feature, value in fv.items():
            row.at[feature] = value
        return row

    data = data.apply(compute_mean_features, axis=1).copy()
    # three patients have a missing T2 and thus many NAs, we remove them.
    data = data[~data['Patient'].isin(["BNI0059", "MCH2978_1", "MCH2989"])]

    data["ID"] = data["Patient"]
    data["Target"] = data["EGFRAlt"] == 1.0
    data = data.drop(['Unnamed: 0', 'Patient', 'X', 'Y', 'Z', 'ENH.1.BAT.0',
       'T1.window.in.T2..1..or.not..0.', 'GenderM.1.F.0.',
       'Recurrent.1.Primary.0', 'EGFRAlt', 'PDGFRAAlt', 'PTENAlt', 'exists'], axis = 1)
    return data.copy()



folder = './raw/Zhang2024A'
def Zhang2024A (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "Baseline characteristics of patients.csv"
    meta = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")

    inputFile = "CT  radiomics features.csv"
    ct = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
    ct.columns = ["ID"] + [f"CT_{k}" for k in ct.keys() if k != "ID"]

    inputFile = "PET  radiomics features.csv"
    pet = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
    pet.columns = ["ID"] + [f"PET_{k}" for k in pet.keys() if k != "ID"]
    meta.shape
    merged_df = pd.merge(meta, ct, on='ID', how='left')
    merged_df = pd.merge(merged_df, pet, on='ID', how='left')
    merged_df["Target"] = merged_df["histology"]
    merged_df = merged_df.drop(['gender', 'age', 'BMI', 'smoking', 'T stage', 'N stage', 'stage',
       'WBC', 'NEU', 'LYM', 'EOS', 'BAS', 'PLT', 'CEA', 'LDH', 'ALB', 'Ca2+',
       'NLR', 'TLG', 'SUVmean', 'MTV', 'SUVmax', 'SUVmin', 'dNLR', 'histology'], axis = 1).copy()

    return merged_df.copy()




folder = './raw/Dong2022'
def Dong2022 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "raw_data.csv"
    data = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")

    # label = 1 --> TBG group
    data["ID"] = data["Unnamed: 0"]
    data["Target"] = data["label"]
    data = data.drop(['label', 'Unnamed: 0'], axis = 1)
    # c5 is duplicated it seems
    data.drop_duplicates(subset="ID", inplace=True)
    return data.copy()



folder = './raw/Zhu2023'
def Zhu2023 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "raw_data.xlsx"
    data = pd.read_excel(os.path.join(dataDir, inputFile))
    data['Target'] = data["lymph"]
    data = data.iloc[:,36:].copy()
    data = data.reset_index(drop = True).reset_index(drop = False)
    data["ID"] = data['index']
    data = data.drop(['index'], axis = 1)
    return data.copy()



folder = './raw/Zhang2024B'
def Zhang2024B (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "train.csv"
    dataTr = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")

    inputFile = "valiad.csv"
    dataVal = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")

    data = pd.concat([dataTr, dataVal], axis = 0)
    data["Target"] = data["N10"]
    data["ID"] = data["Unnamed: 0"]
    data = data.iloc[:,10:].copy()
    return data.copy()



folder = './raw/Dai2023'
def Dai2023 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "radiomics_features.xlsx"
    data = pd.read_excel(os.path.join(dataDir, inputFile))
    data["Target"] = data["V1"]
    data["ID"] = data["Unnamed: 0"]
    data = data.drop(['Unnamed: 0', 'V1'], axis = 1)
    return data.copy()




folder = './raw/Zhang2023'
def Zhang2023 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "radiomics_data.txt"
    data = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
    data["Target"] = data["Lable1"]
    data = data.reset_index(drop = True).reset_index(drop = False)
    data["ID"] = data['index']
    data = data.drop(['index', 'Lable1'], axis = 1)
    return data.copy()



folder = './raw/Fusco2022'
def Fusco2022 (folder = None):
    dataDir = os.path.join(folder, "data")
    inputFile = "All_features - Copia.csv"
    data = pd.read_csv(os.path.join(dataDir, inputFile), sep=";")
    data.drop_duplicates(subset='ID_paziente', keep='first', inplace=True)
    data["ID"] = data["ID_paziente"]
    data["Target"] = data["Istologico"] == "POSITIVO"
    data = data.drop(["ID_paziente", "ID_lesione", "Birads", "Istologico"], axis = 1)
    return data.copy()



def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as f:
        yaml.dump(dictionary, f, default_flow_style=False)



def scaleData (data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)


def normalizeData (data):
    # imputation does not make any sense for columns with nearly all NA
    IDs, targets = data["ID"], data["Target"]
    X = data.drop(["ID", "Target"], axis = 1).copy()

    nNaBefore = X.isna().sum().sum()

    removeNACols = list(X.keys()[ (X.isna().sum(axis = 0) >= X.shape[0]//4) ])
    for k in removeNACols:
        X[k] = np.random.normal(0,1,X.shape[0])
    nReplacedFeatures = len(removeNACols)

    simp = SimpleImputer(strategy="mean")
    X = pd.DataFrame(simp.fit_transform(X),columns = X.columns)
    nNaAfter = X.isna().sum().sum()
    assert (nNaAfter == 0)

    # fix constant variables to be random, which reduces problems later
    np.random.seed(471)
    dropKeys = [z for z in X.keys() if len(set(X[z].values))==1]
    for k in dropKeys:
        X[k] = np.random.normal(0,1,X.shape[0])
    nReplacedFeatures = nReplacedFeatures + len(dropKeys)

    X = scaleData (X)
    X["ID"], X["Target"] = IDs, targets

    assert (data.shape == X.shape)
    print (f"\t{nNaBefore} missings, {nReplacedFeatures} replaced features")
    return X, nNaBefore, nReplacedFeatures





if __name__ == '__main__':
    datasets = {}
    table1 = []

    #recreatePath ('./datasets')
    print ("\nProcessing datasets...")
    metadata = {}
    for f in sorted(glob("./raw/*/*.yaml"))[::-1]:
        # special treatment
        # if "LIDC-IDRI" not in f:
        #     continue
        print("\t", f)
        if "radiomics.uk" in f:
            dset = os.path.basename(f).split("_")[0]
            # we need to sort out
            if dset not in radiomics_uk_datasets:
                continue
            z = RadTabularDataLoader (dset, folder = "./raw/radiomics.uk")
            data = z.data
        else:
            dset = os.path.basename(os.path.dirname(f))
            exec (f'data = {dset}("{os.path.dirname(f)}")')

        data = data.reset_index(drop = True) # ensure the index is normalized, we have IDs anyway
        datadim = float(np.round(data.shape[1]/data.shape[0],2)) # e.g. 2.0 = twice as many features as samples
        databal = int(np.round(np.sum(data["Target"])/len(data)*100,0))
        data, nNaBefore, nReplacedFeatures = normalizeData (data)
        checkData (data)

        data.to_csv(f"./datasets/{dset}.gz", compression = "gzip", index = False)
        y = getMetadata(dset, folder = "./raw")
        y["nInstances"] = data.shape[0]
        y["nFeatures"] = data.shape[1] - 2 # because of Target and ID
        y["Dimensionality"] = datadim
        y["ClassBalance"] = databal # as frequency of the positive class
        pMissing = int(nNaBefore) # direct number of missing values
        y["Missings"] = pMissing
        y["BrokenFeatures"] = nReplacedFeatures
        metadata[dset] = y

    save_dict_to_file (metadata, "./radMLBench/metadata.yaml")

 #
