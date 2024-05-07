#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(
    name='radMLBench',
    version='1.0',
    author='Aydin Demircioglu',
    author_email=('aydin.demircioglu@uk-essen.de'),
    packages=find_packages(),
    package_data={'radMLBench': ['./metadata.yaml']},
    include_package_data=True,
    url='https://github.com/aydindemircioglu/radMLBench',
    license="MIT",
    description=('A Python wrapper for the radMLBench data repository.'),
    long_description='''
A Python wrapper for the radMLBench data repository.

Contact
=============
If you have any questions or comments about radMLBench,
please feel free to contact us via e-mail: aydin.demircioglu@uk-essen.de

This project is hosted at https://github.com/aydindemircioglu/radMLBench
''',
    zip_safe=True,
    install_requires=['pandas>=2.0.0',
                    'requests>=2.24.0',
                    'pyyaml>=5.3.1',
                    'joblib>=1.2.0',
                    'numpy>=1.10.0',
                    'scikit-learn>=1.1.0'
                    ],
    python_requires=">=3.6.0", # guess
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=['radiomics', 'data mining', 'benchmark', 'machine learning', 'data analysis', 'data sets', 'data science', 'wrapper'],
)
