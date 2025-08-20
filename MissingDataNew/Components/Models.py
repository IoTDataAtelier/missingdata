from pypots.imputation import SAITS, BRITS, USGAN, GPVAE, MRNN
from pypots.optim import Adam
import numpy as np

class Models:

    def model(model, dataset):

        if model == "saits":
            model_imputation= SAITS(
                n_steps=dataset['n_steps'],
                n_features=dataset['n_features'],
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
        
        elif model == "brits":
            model_imputation = BRITS(
                n_steps=dataset['n_steps'],
                n_features=dataset['n_features'],
                rnn_hidden_size=128,
                batch_size=32,
                epochs=10,
                patience=3,
                optimizer=Adam(lr=1e-3),
                num_workers=0,
                device=None,
                model_saving_strategy="best",
            )

        elif model == "usgan":
            
            model_imputation = USGAN(
                n_steps=dataset['n_steps'],
                n_features=dataset['n_features'],
                rnn_hidden_size=256,
                lambda_mse=1,
                dropout=0.1,
                G_steps=1,
                D_steps=1,
                batch_size=32,
                epochs=10,
                patience=3,
                G_optimizer=Adam(lr=1e-3),
                D_optimizer=Adam(lr=1e-3),
                num_workers=0,
                device=None,
                model_saving_strategy="best",
            )
        
        elif model == "gpvae":
            model_imputation = GPVAE(
                n_steps=dataset['n_steps'],
                n_features=dataset['n_features'],
                latent_size=37,
                encoder_sizes=(128,128),
                decoder_sizes=(256,256),
                kernel="cauchy",
                beta=0.2,
                M=1,
                K=1,
                sigma=1.005,
                length_scale=7.0,
                kernel_scales=1,
                window_size=24,
                batch_size=32,
                epochs=10,
                patience=3,
                optimizer=Adam(lr=1e-3),
                num_workers=0,
                device=None,
                model_saving_strategy="best",
            )
        
        elif model == "mrnn":
            model_imputation = MRNN(
                n_steps=dataset['n_steps'],
                n_features=dataset['n_features'],
                rnn_hidden_size=128,
                epochs=10,
                patience=3,
                optimizer=Adam(lr=1e-3),
                num_workers=0,
                device=None,
                model_saving_strategy="best",
            )

        return model_imputation
    
    def train_load_model(model, dataset_training, dataset_validation, train, path = ""):
        if train == True:
            model.fit(dataset_training, dataset_validation)
        else:
            model.load(path)

    def imputation(model, dataset_testing, path_save = ""):
        imputed_dataset = model.predict(dataset_testing)
        imputed_dataset = imputed_dataset["imputation"]
        np.save(path_save, imputed_dataset)
        return imputed_dataset


        
        
