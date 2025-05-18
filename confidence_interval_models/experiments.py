import os
import sys
import pypots
import numpy as np
import benchpots
import torch
import matplotlib.pyplot as plt
from pypots.optim import Adam
from pypots.imputation import SAITS, BRITS, USGAN, GPVAE, MRNN
from pypots.utils.random import set_random_seed
from multiprocessing import Process, set_start_method
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

#from MAEModify.error import calc_mae



def experiments():

    set_random_seed()
    physionet2012_dataset = benchpots.datasets.preprocess_physionet2012(subset="all", rate=0.1)

    dataset_for_training = {
        "X": physionet2012_dataset['train_X'],
    }

    dataset_for_validating = {
        "X": physionet2012_dataset['val_X'],
        "X_ori": physionet2012_dataset['val_X_ori'],
    }

    dataset_for_testing = {
        "X": physionet2012_dataset['test_X'],
    }

    test_X_indicating_mask = np.isnan(physionet2012_dataset['test_X_ori']) ^ np.isnan(physionet2012_dataset['test_X'])
    test_X_ori = np.nan_to_num(physionet2012_dataset['test_X_ori']) 

    saits = SAITS(
        n_steps=physionet2012_dataset['n_steps'],
        n_features=physionet2012_dataset['n_features'],
        n_layers=1,
        d_model=256,
        d_ffn=128,
        n_heads=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        ORT_weight=1,  
        MIT_weight=1,
        batch_size=32,
        epochs=10,
        patience=3,
        optimizer=Adam(lr=1e-3),
        num_workers=0,
        device=None,
        model_saving_strategy="best",
    )

    saits_results = saits.predict(dataset_for_testing)
    saits_imputation = saits_results["imputation"]

    saits.load("../mae/tutorial_results/imputation/saits/20250422_T181642/SAITS.pypots")

    saits_mae = pypots.calc_mae(
        saits_imputation,
        test_X_ori,
        test_X_indicating_mask,
    )

    print(saits_mae)


with torch.cuda.device(0):
    if __name__ == '__main__':
        set_start_method('spawn')
        info('main line')
        p = Process(target=experiments)
        p.start()
        p.join() ## entrando