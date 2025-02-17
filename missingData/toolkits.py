from pypots.utils.random import set_random_seed
from pypotsModify.benchpotsMAE.datasets import preprocess_physionet2012

class toolkits:
    def load_dataset(subset, rate, normalization):
        set_random_seed()
        physionet2012_dataset = preprocess_physionet2012(subset=subset, rate=rate, normalization=normalization)
        return physionet2012_dataset
    
    def separaring_dataset(dataset):
        dataset_for_training = {
            "X": dataset['train_X'],
        }
        