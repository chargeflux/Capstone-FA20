import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})
import CustomNMF

import surprise
from surprise import Dataset
from surprise import Reader
import pandas as pd
from sklearn import preprocessing
from functools import partial

def scale_cycles(input_data):
    '''
    Apply log transformation then MinMaxScaler from 0 to 1 on cycle count
    '''
    input_data.cycles = np.log(input_data.cycles)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    cycles_scaled_min_max = min_max_scaler.fit_transform(input_data.cycles.values.reshape(-1,1))
    input_data["cycles"] = cycles_scaled_min_max

    return input_data

def get_best_config(input_data, unknown_program_name):
    model = CustomNMF.NMF(biased=True, n_epochs=20, n_factors=3, lr_bu=0.03, lr_bi=0.01,
                          reg_pu=0.01, reg_qi=0.001, reg_bu=0.005, reg_bi=0.05)

    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(input_data, reader)

    model.fit(data.build_full_trainset())

    estimates = {}
    for possible_config in input_data.config.unique():
        est = model.predict(uid=unknown_program_name, iid=possible_config).est
        estimates[possible_config] = est

    best_config = sorted(estimates.items(), key= lambda x: x[1])[0][0]
    return best_config.split(",")

