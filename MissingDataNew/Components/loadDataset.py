import benchpots
import numpy as np
from pypots.utils.random import set_random_seed

class loadDataset:

    #Carrega dataset escolhido
    def load_dataset_pypots(dataset, subset, rate):
        set_random_seed()

        if dataset == "physionet_2012":
            dataset_load = benchpots.datasets.preprocess_physionet2012(subset, rate)
        
        print(dataset_load.keys())

        return dataset_load
    
    #Separa o dataset em treino, validação e test
    def separating_dataset(dataset):
        dataset_for_train = {
            "X": dataset["train_X"]
        }

        dataset_for_validation = {
            "X": dataset["val_X"],
            "X_ori": dataset["val_X_ori"]
        }

        dataset_for_testing = {
            "X": dataset["test_X"]
        }

        return dataset_for_train, dataset_for_validation, dataset_for_testing
    
    #Cria a indicating mask para o test
    def create_indicating_mask(dataset_testing_ori, dataset_testing):
        indicating_mask = np.isnan(dataset_testing_ori) ^ np.isnan(dataset_testing)

        return indicating_mask
    
    #Tranforma nan em zero no dataset
    def transform_nan_to_zero(dataset):
        
        nan_to_zero = np.nan_to_num(dataset)

        return nan_to_zero