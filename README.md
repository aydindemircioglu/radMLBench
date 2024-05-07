# radMLBench

This is a large collection of radiomic datasets, accessible via Python.


## Installing radMLBench

Install it via pip:

```
pip install radMLBench
```


## Datasets

All data sets are stored in a common format:

* First row is the column names
* Each following row corresponds to one row of the data
* The ID column is named `ID` (in case the original data did not supply IDs, these are simple numbers)
* The target column is named `Target`, always binary (=0,1)
* All files are compressed with `gzip` to conserve space


The complete list of datasets is

<!-- dataset table:start -->

| Dataset                     |   Year | Modality   |   #Instances |   #Features |   Dimensionality |   ClassBalance | DOI                                          |
|:----------------------------|-------:|:-----------|-------------:|------------:|-----------------:|---------------:|:---------------------------------------------|
| Ahn2021                     |   2021 | MRI        |          114 |         108 |             0.96 |             55 | https://doi.org/10.1016/j.ejrad.2019.108642  |
| Arita2018                   |   2018 | MRI        |          168 |         684 |             4.08 |             66 | https://doi.org/10.1038/s41598-018-30273-4   |
| BraTS-2021                  |   2021 | MRI        |          577 |        4060 |             7.04 |             52 | https://doi.org/10.48550/arXiv.2107.02314    |
| Brancato2023                |   2023 | MRI        |           58 |        2380 |            41.07 |             60 | https://doi.org/10.3390/jcm12010140          |
| C4KC-KiTS                   |   2021 | CT         |           70 |         315 |             4.53 |             66 | https://doi.org/10.1016/j.media.2020.101821  |
| Colorectal-Liver-Metastases |   2024 | CT         |           90 |         525 |             5.86 |             84 | https://doi.org/10.1038/s41597-024-02981-2   |
| Dai2023                     |   2023 | CT         |          119 |         851 |             7.17 |             26 | https://doi.org/10.7717/peerj.16230          |
| Deng2023                    |   2023 | MRI        |          261 |         224 |             0.87 |             36 | https://doi.org/10.1007/s13246-023-01300-0   |
| Dong2022                    |   2022 | CT         |          279 |         851 |             3.06 |             49 | https://doi.org/10.7717/peerj.14127          |
| Fusco2022                   |   2022 | MRI        |           54 |         192 |             3.59 |             61 | https://doi.org/10.3390/curroncol29030159    |
| Granata2021                 |   2021 | CT         |           88 |         580 |             6.61 |             47 | https://doi.org/10.3390/cancers13163992      |
| Granata2024                 |   2024 | MRI        |           51 |         851 |            16.73 |             75 | https://doi.org/10.3390/diagnostics14020152  |
| HCC-TACE-Seg                |   2023 | CT         |           84 |         420 |             5.02 |             14 | https://doi.org/10.1038/s41597-023-01928-3   |
| HNSCC                       |   2018 | CT         |           93 |         105 |             1.15 |             27 | https://doi.org/10.1038/sdata.2018.173       |
| Head-Neck-PET-CT            |   2017 | PET/CT     |           91 |         210 |             2.33 |             67 | https://doi.org/10.1038/s41598-017-10371-5   |
| Head-Neck-Radiomics-HN1     |   2014 | CT         |          137 |         105 |             0.78 |             45 | http://doi.org/10.1038/ncomms5006            |
| Hosny2018A                  |   2018 | CT         |          293 |         984 |             3.37 |             54 | https://doi.org/10.1371/journal.pmed.1002711 |
| Hosny2018B                  |   2018 | CT         |          207 |         984 |             4.76 |             29 | https://doi.org/10.1371/journal.pmed.1002711 |
| Hosny2018C                  |   2018 | CT         |          183 |         984 |             5.39 |             73 | https://doi.org/10.1371/journal.pmed.1002711 |
| Huang2023                   |   2023 | CT         |          212 |         855 |             4.04 |             46 | https://doi.org/10.1371/journal.pone.0292110 |
| Hunter2023                  |   2023 | CT         |          520 |        1998 |             3.85 |             54 | https://doi.org/10.1038/s41416-023-02480-y   |
| ISPY1                       |   2016 | MRI        |          161 |         370 |             2.31 |             57 | http://doi.org/10.7937/K9/TCIA.2016.HdHpgJLK |
| Keek2020                    |   2020 | CT         |          273 |        1322 |             4.85 |             44 | https://doi.org/10.1371/journal.pone.0232639 |
| LGG-1p19qDeletion           |   2017 | MRI        |          159 |        2030 |            12.78 |             64 | https://doi.org/10.1007/s10278-017-9984-3    |
| LNDb                        |   2019 | CT         |          173 |         105 |             0.62 |             66 | https://doi.org/10.48550/arXiv.1911.08434    |
| Li2020                      |   2020 | MRI        |           51 |         396 |             7.8  |             63 | https://doi.org/10.1371/journal.pone.0227703 |
| Lu2019                      |   2019 | CT         |           75 |         657 |             8.79 |             73 | https://doi.org/10.1038/s41467-019-08718-9   |
| Meningioma-SEG-CLASS        |   2022 | MRI        |           88 |        2030 |            23.09 |             43 | https://doi.org/10.1038/s41598-022-07859-0   |
| NSCLC-Radiogenomics         |   2012 | PET/CT     |          144 |         105 |             0.74 |             16 | http://doi.org/10.1148/radiol.12111607       |
| OcanaTienda2023             |   2023 | MRI        |           67 |        1130 |            16.9  |             48 | https://doi.org/10.1038/s41597-023-02123-0   |
| PI-CAI                      |   2021 | MRI        |          969 |        3045 |             3.14 |             66 | https://doi.org/10.1016/j.media.2021.102155  |
| Petrillo2023                |   2023 | MRI        |          128 |         851 |             6.66 |             37 | https://doi.org/10.1007/s11547-023-01718-2   |
| Prostate-MRI-US-Biopsy      |   2013 | MRI        |          773 |        1015 |             1.32 |             77 | https://doi.org/10.1016/j.juro.2012.08.095   |
| QIN-HEADNECK                |   2016 | PET/CT     |           59 |         210 |             3.59 |             75 | https://doi.org/10.7717/peerj.2057           |
| Ramella2018                 |   2018 | CT         |           91 |         242 |             2.68 |             55 | https://doi.org/10.1371/journal.pone.0207455 |
| Sasaki2019                  |   2019 | MRI        |          138 |         587 |             4.27 |             49 | https://doi.org/10.1038/s41598-019-50849-y   |
| Song2020                    |   2020 | MRI        |          260 |         264 |             1.02 |             49 | https://doi.org/10.1371/journal.pone.0237587 |
| UCSF-PDGM                   |   2022 | MRI        |          418 |        7105 |            17    |             89 | https://doi.org/10.1148/ryai.220058          |
| UPENN-GBM                   |   2022 | MRI        |          187 |       11165 |            59.72 |             42 | https://doi.org/10.1038/s41597-022-01560-7   |
| Veeraraghavan2020           |   2020 | MRI        |          150 |         200 |             1.35 |             31 | https://doi.org/10.1038/s41598-020-72475-9   |
| WORC-CRLM                   |   2021 | CT         |           77 |        1015 |            13.21 |             48 | https://doi.org/10.48550/arXiv.2108.08618    |
| WORC-Desmoid                |   2021 | MRI        |          203 |        1015 |             5.01 |             35 | https://doi.org/10.48550/arXiv.2108.08618    |
| WORC-GIST                   |   2021 | CT         |          245 |        1015 |             4.15 |             51 | https://doi.org/10.48550/arXiv.2108.08618    |
| WORC-Lipo                   |   2021 | MRI        |          114 |        1015 |             8.92 |             50 | https://doi.org/10.48550/arXiv.2108.08618    |
| WORC-Liver                  |   2021 | MRI        |          186 |        1015 |             5.47 |             51 | https://doi.org/10.48550/arXiv.2108.08618    |
| WORC-Melanoma               |   2021 | CT         |           95 |        1015 |            10.71 |             49 | https://doi.org/10.48550/arXiv.2108.08618    |
| Wang2024                    |   2024 | MRI        |           67 |         280 |             4.21 |             40 | https://doi.org/10.1371/journal.pone.0299267 |
| Zhang2023                   |   2023 | CT         |          203 |        1781 |             8.78 |             51 | https://doi.org/10.7717/peerj.14559          |
| Zhang2024A                  |   2024 | PET/CT     |          255 |        3850 |            15.11 |             57 | https://doi.org/10.1371/journal.pone.0300170 |
| Zhang2024B                  |   2024 | CT         |          192 |         833 |             4.35 |             66 | https://doi.org/10.7717/peerj.17111          |
<!-- dataset table:end -->


## Python wrapper

For easy access to the benchmark data sets, we have provided a Python wrapper named `radMLBench`. The wrapper can be installed on Python via `pip`:

```
pip install radMLBench
```

and used in Python scripts as follows:

```python
from radMLBench import listDatasets, getMetaData, loadData

for dataset in radMLBench.listDatasets():
  print (f"Loading {dataset}")
  data = radMLBench.loadData(dataset)
  print (data.shape)
```

There is a simple example on how to use it in combination with a random forest
classifier in `./examples/simpleTest.py`.

Ideally, one should download the data to a local directory so that
downloading from the internet is minimized for speed reaons. To do so,
just use the `local_cache_dir` variable:

``` data = radMLBench.loadData("Wang2024", local_cache_dir="~/radMLBench.repo") ```

In case you want to load the data not as a dataframe (where ID and Target columns
exist), but as a ready-to-use numpy dataset, just add the parameter `return_X_y`.
This will convert the pandas dataframe into two numpy arrays X, and y. X will
contain the data and y the labels.

``` data = radMLBench.loadData("Wang2024", local_cache_dir="~/radMLBench.repo", return_X_y = True) ```



## Experiments

The decorrelation example can be found in `examples/decorrelation`,
to execute it, start `./examples/decorrelation/decorrelation.py`.

The other example, TabFPN, can be found in `examples/tebpfn`.
However, before executing this, one must install TabFPN (which in turn
will install things like torch). One must also execute TabFPN once
before  executing the script, since TabFPN must first download the model--
but this will not work if multiple TabFPN instances try it at the same time.
Therefore, one must first (with a single CPU core) execute TabFPN.
One can achieve this by setting downloadFirst to True in the script.
After executing the script (which will just download and exit), one can
put downloadFirst to False again to execute the experiment.




## Citing radMLBench

If you use radMLBench in a scientific publication, please consider citing the paper:

Aydin Demircioglu.
[radMLBench: A radiomics dataset collection for benchmarking in radiomics](TBD).
_TBD_ (2024).

(This will update as soon as the manuscript is submitted).

```
