# -*- coding: utf-8 -*-

"""
radMLBench was developed at the University Hospital in Essen, Germany.
For questions use github issues or write an email (aydin.demircioglu@uk-essen.de).

radMLBench is partially based on the PMLB (Penn Machine Learning Benchmarks), see https://epistasislab.github.io/pmlb/

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import pandas as pd
import numpy as np
import os
from pkg_resources import resource_filename
import requests
import warnings
import subprocess
import pathlib
import yaml

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np

GITHUB_URL = 'https://github.com/aydindemircioglu/radMLBench/raw/main/datasets'
suffix = '.gz'
metaData = None



def listDatasets (sort_by = None):
    """Return a list of all available datasets

    Parameters
    ----------
    sort_by: str
        The column to sort by. If column does not exist, the alphabetically sorted list will be returned.

    Returns
    ----------
    datasets: list
        The list of all available datasets, sorted by sort_by.
    """
    # ensure its loaded
    global metaData
    readMetaData()
    dsets = list(metaData.keys())
    try:
        if sort_by is not None:
            dsets = sorted(metaData.keys(), key=lambda x: metaData[x][sort_by])
    except:
        pass
    return dsets



def get_dataset_url(GITHUB_URL, dataset_name, suffix):
    dataset_url = '{GITHUB_URL}/{DATASET_NAME}{SUFFIX}'.format(
                                GITHUB_URL=GITHUB_URL,
                                DATASET_NAME=dataset_name,
                                SUFFIX=suffix
                                )

    re = requests.get(dataset_url)
    if re.status_code != 200:
        raise ValueError(f'Unable to retrieve dataset from {dataset_url}')
    return dataset_url



def loadData (dataset_name,return_X_y=False, local_cache_dir=None):
    """Download a data set from the radMLBench, (optionally) store it locally, and return the data set.

    You must be connected to the internet if you are fetching a data set that is not cached locally.

    Parameters
    ----------
    dataset_name: str
        The name of the data set to load from PMLB.
    return_X_y: bool (default: False)
        Whether to return the data in scikit-learn format, with the features
        and labels stored in separate NumPy arrays.
    local_cache_dir: str (default: None)
        The directory on your local machine to store the data files.
        If None, then the local data cache will not be used.

    Returns
    ----------
    dataset: pd.DataFrame or (array-like, array-like)
        if return_X_y == False: A pandas DataFrame containing the fetched data set (with columns ID and Target)
        if return_X_y == True: A tuple of NumPy arrays containing (features, labels)
    """

    if dataset_name not in listDatasets():
        raise ValueError('Dataset not found?')

    if local_cache_dir is None:
        dataset_url = get_dataset_url(GITHUB_URL, dataset_name, suffix)
        dataset = pd.read_csv(dataset_url, compression='gzip')
    else:
        dataset_path = os.path.join(local_cache_dir, dataset_name+suffix)
        if os.path.exists(dataset_path):
            dataset = pd.read_csv(dataset_path, compression='gzip')
        else:
            print (f"Downloading to {dataset_path}...")
            dataset_url = get_dataset_url(GITHUB_URL, dataset_name, suffix)
            dataset = pd.read_csv(dataset_url, compression='gzip')
            dataset_dir = os.path.split(dataset_path)[0]
            if not os.path.isdir(dataset_dir):
                os.makedirs(dataset_dir)
            dataset.to_csv(dataset_path, compression='gzip', index=False)

    if return_X_y:
        X = dataset.drop(['ID', 'Target'], axis=1).values
        y = dataset['Target'].values
        return (X, y)
    else:
        return dataset



def getCVSplits(dataset, num_splits=10, num_repeats=10):
    """
    Generate indices for cross-validation splits.

    Parameters:
    - dataset (str or DataFrame or tuple of numpy arrays): Input dataset. If str, it will be treated as dataset name to load.
    - num_splits (int): Number of folds for cross-validation.
    - num_repeats (int): Number of times to repeat the cross-validation process.

    Returns:
    - List of tuples: Each tuple contains train and test indices for one split.
    """
    if isinstance(dataset, str):
        dataset = loadDataset(dataset)

    random_state = 42 * num_splits + num_repeats + 42

    if isinstance(dataset, pd.DataFrame):
        y = dataset["Target"].values
        X = dataset.drop(columns=["Target", "ID"]).values
    else:
        X, y = dataset

    cv_splits = []
    rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=random_state)
    for train_index, test_index in rskf.split(X, y):
        cv_splits.append((train_index, test_index))

    return cv_splits



def readMetaData ():
    """Internal. Reads metadata into variable"""
    def read_yaml(file_path):
        try:
            with open(file_path, 'r') as stream:
                data = yaml.safe_load(stream)
                return data
        except Exception as e:
            print ('Unable to read metadata file for all dataset!')
            print ('Error', e)
            return None

    global metaData
    if metaData is None:
        #print ("Reading metadata.")
        package_dir = resource_filename('radMLBench', '')
        metadata_path = os.path.join(package_dir, 'metadata.yaml')
        metaData = read_yaml(metadata_path)
    pass




def getMetaData (dataset_name):
    """Retrieve the metadata for a data set from the radMLBench.

    Parameters
    ----------
    dataset_name: str
        The name of the data set to retrieve the metadata for.

    Returns
    ----------
    metadata: dictionary
        Dictionary containing the metadata
    """
    # ensure its loaded
    global metaData
    readMetaData()
    if dataset_name not in metaData:
        print ("Unknown dataset.")
        return None
    return metaData[dataset_name]



#
