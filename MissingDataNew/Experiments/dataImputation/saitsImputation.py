import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from Components.loadDataset import loadDataset as ld
from Components.Models import Models as md 

#Carrega o dataset physionet
physionet2012_dataset = ld.load_dataset_pypots("physionet_2012", "all", 0.1)

#Separa o dataset physionet em treino, validação e teste 
dataset_for_training, dataset_for_validating, dataset_for_testing = ld.separating_dataset(physionet2012_dataset)

#Cria o indicating mask para o test
test_X_indicating_mask = ld.create_indicating_mask(physionet2012_dataset["test_X_ori"], physionet2012_dataset["test_X"])

#Tranforma os nan do dataset em zero
test_X_ori = ld.transform_nan_to_zero(physionet2012_dataset["test_X_ori"])

#Cria a instância do modelo com seus parâmetros
saits = md.model("saits", physionet2012_dataset)

#Carrega treinamento do modelo existente
path = "/data/victor/missingdata/MissingDataNew/Components/TrainedModels/saits/20250422_T181642/SAITS.pypots"
md.train_load_model(saits, dataset_for_training, dataset_for_testing, False, path)

#Realiza a imputação e salva o dataset imputado
path_save_imputation = "/data/victor/missingdata/MissingDataNew/ImputedDatasets/Physionet2012/physionet_saits.npy"
saits_imputation = md.imputation(saits, dataset_for_testing, path_save_imputation)

