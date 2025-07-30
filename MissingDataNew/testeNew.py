import os
import numpy as np
import benchpots
import torch
from pypots.utils.random import set_random_seed
from pypots.nn.functional import calc_mae
from pypots.optim import Adam
from pypots.imputation import SAITS
from multiprocessing import Process, set_start_method


#Load Physionet
set_random_seed()
physionet2012_dataset = benchpots.datasets.preprocess_physionet2012(
    subset="set-a",
    rate=0.1
)

dataset_for_IMPU_training = {
    "X": physionet2012_dataset['train_X'],
}

dataset_for_IMPU_validating = {
    "X": physionet2012_dataset['val_X'],
    "X_ori": physionet2012_dataset['val_X_ori'],
}

dataset_for_IMPU_testing = {
    "X": physionet2012_dataset['test_X'],
}

test_X_indicating_mask = np.isnan(physionet2012_dataset['test_X_ori']) ^ np.isnan(physionet2012_dataset['test_X'])
test_X_ori = np.nan_to_num(physionet2012_dataset['test_X_ori']) 

def saitsExperiments():
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
    saving_path="tutorial_results/imputation/saits",
    model_saving_strategy="best",
    )
    
    saits.fit(
        train_set = dataset_for_IMPU_training,
        val_set=dataset_for_IMPU_validating
    )

    saits_results = saits.predict(dataset_for_IMPU_testing)
    saits_imputation = saits_results["imputation"]

    testing_mae = calc_mae(
        saits_imputation,
        test_X_ori,
        test_X_indicating_mask,
    )

    print(f"Testing mean absolute error: {testing_mae:.4f}")

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


with torch.cuda.device(1):
    if __name__ == '__main__':
        set_start_method('spawn')
        info('main line')
        p = Process(target=saitsExperiments)
        p.start()
        p.join()

saitsExperiments()