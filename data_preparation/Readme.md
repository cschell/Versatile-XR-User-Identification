This folder contains the scripts required to preprocess the ['Who is Alyx?'](https://github.com/cschell/who-is-alyx) dataset for training and evaluation of the classification- and embedding-based models.

# Preparation

1. clone the dataset from https://github.com/cschell/who-is-alyx into `data/input`
2. if you want to use the same database as used in the paper, you manually have to delete the folders of the last 8 players, as their data was added after work on the paper had begun
3. install python requirements with `pip install -r requirements.txt`

# Usage

1. `python 01_aggregate.py` produces an intermediate file used by the following scripts
2. `python 02.1_generate_classifier_dataset.py` produces the HDF5 file for the classification-based model
3. `python 02.2_generate_dml_dataset.py` produces the HDF5 file for the embedding-based model

The HDF5 files in `data/output` are used for training the machine learning models. Make sure to edit the path in the hydra config files.