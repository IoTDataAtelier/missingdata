import numpy as np

class Results:

    def reshape_for_patients(model_aes):
        dataset_imputed_reshape = model_aes.reshape(len(model_aes), 48 * 37)
        return dataset_imputed_reshape
    
    def sum_aes(model_aes):
        model_ae_sum  = []

        for model_ae in model_aes:
            model_ae_sum.append(np.sum(model_ae))
        
        return model_ae_sum
    
    def ae_mask(model_aes, indicating_mask):
        model_ae_mask = []
        model_ae_mask_aux = []

        for i in range(len(indicating_mask)):
            for j in range(len(indicating_mask[i])):
                if indicating_mask[i][j] == True:
                    model_ae_mask_aux.append(model_aes[i][j])

            model_ae_mask.append(model_ae_mask_aux)
            model_ae_mask_aux = []


        return model_ae_mask

