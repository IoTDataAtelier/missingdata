import numpy as np
import pypots
import pandas as pd
from pypots.utils.metrics import calc_mae

class toolkits:

    def separating_dataset(dataset):

        dataset_for_training = {
            "X": dataset['train_X'],
        }

        dataset_for_validating = {
            "X": dataset['val_X'],
            "X_ori": dataset['val_X_ori']
        }

        dataset_for_testing_ori = {
            "X_ori": dataset['test_X_ori'],
            "female_gender_test_X_ori": dataset['female_gender_test_X_ori'],
            "male_gender_test_X_ori": dataset['male_gender_test_X_ori'],
            "undefined_gender_test_X_ori": dataset['undefined_gender_test_X_ori'],
            "more_than_or_equal_to_65_test_X_ori":  dataset['more_than_or_equal_to_65_test_X_ori'],
            "less_than_65_test_X_ori": dataset['less_than_65_test_X_ori'],
            "ICUType_1_test_X_ori": dataset['ICUType_1_test_X_ori'],
            "ICUType_2_test_X_ori": dataset['ICUType_2_test_X_ori'],
            "ICUType_3_test_X_ori": dataset['ICUType_3_test_X_ori'],
            "ICUType_4_test_X_ori": dataset['ICUType_4_test_X_ori'],
            "classificacao_undefined_test_X_ori": dataset['classificacao_undefined_test_X_ori'],
            "classificacao_baixo_peso_test_X_ori": dataset['classificacao_baixo_peso_test_X_ori'],
            "classificacao_normal_peso_test_X_ori": dataset['classificacao_normal_peso_test_X_ori'],
            "classificacao_sobrepeso_test_X_ori": dataset['classificacao_sobrepeso_test_X_ori'],
            "classificacao_obesidade_1_test_X_ori": dataset['classificacao_obesidade_1_test_X_ori'],
            "classificacao_obesidade_2_test_X_ori": dataset['classificacao_obesidade_2_test_X_ori'],
            "classificacao_obesidade_3_test_X_ori": dataset['classificacao_obesidade_3_test_X_ori']
        }

        dataset_for_testing = {
            "X": dataset['test_X'],
            "female_gender_test_X": dataset['female_gender_test_X'],
            "male_gender_test_X": dataset['male_gender_test_X'],
            "undefined_gender_test_X": dataset['undefined_gender_test_X'],
            "more_than_or_equal_to_65_test_X":  dataset['more_than_or_equal_to_65_test_X'],
            "less_than_65_test_X": dataset['less_than_65_test_X'],
            "ICUType_1_test_X": dataset['ICUType_1_test_X'],
            "ICUType_2_test_X": dataset['ICUType_2_test_X'],
            "ICUType_3_test_X": dataset['ICUType_3_test_X'],
            "ICUType_4_test_X": dataset['ICUType_4_test_X'],
            "classificacao_undefined_test_X": dataset['classificacao_undefined_test_X'],
            "classificacao_baixo_peso_test_X": dataset['classificacao_baixo_peso_test_X'],
            "classificacao_normal_peso_test_X": dataset['classificacao_normal_peso_test_X'],
            "classificacao_sobrepeso_test_X": dataset['classificacao_sobrepeso_test_X'],
            "classificacao_obesidade_1_test_X": dataset['classificacao_obesidade_1_test_X'],
            "classificacao_obesidade_2_test_X": dataset['classificacao_obesidade_2_test_X'],
            "classificacao_obesidade_3_test_X": dataset['classificacao_obesidade_3_test_X']
        }

        return dataset_for_training, dataset_for_validating, dataset_for_testing_ori, dataset_for_testing
    
    #Reshape to variable
    def pre_reshape(dataset):
        for i in range(len(dataset)):
            dataset[i] = dataset[i].reshape(len(dataset[i])*48, 37)
        
        return dataset
        
    
    def reshape_variable(dataset):
        listaMed = []
        listaAux = []
        dataset_variable = []

        for i in range(len(dataset)):
            for j in range(37):
                for k in range(len(dataset[i])):
                    listaAux.append(dataset[i][k][j])
                listaMed.append(listaAux)
                listaAux = []
            listaMed = np.array(listaMed)
            dataset_variable.append(listaMed)
            listaMed = []

        return dataset_variable

    #Indicating mask
    def indicating_mask(dataset_for_testing_ori, dataset_for_testing):
        test_X_indicating_mask = []
        test_X_ori = []
        for i, j in zip(dataset_for_testing_ori.values(), dataset_for_testing.values()):
            test_X_indicating_mask.append(np.isnan(i) ^ np.isnan(j))
            test_X_ori.append(np.nan_to_num(i))

        return test_X_indicating_mask, test_X_ori

    def indicating_mask_variable(dataset_for_testing_ori, dataset_for_testing, original, scaler):
        test_X_indicating_mask_variable = []
        test_X_ori_variable = []
        test_X_indicating_mask, test_X_ori = toolkits.indicating_mask(dataset_for_testing_ori, dataset_for_testing)

        if(original == True):
            for i in range(len(test_X_ori)):
                test_X_ori_variable.append(test_X_ori[i].reshape(len(test_X_ori[i])* 48, 37))
                test_X_ori_variable[i] = scaler.inverse_transform(test_X_ori_variable[i])
                test_X_ori_variable[i] = np.nan_to_num(test_X_ori_variable[i])
            
            test_X_indicating_mask_variable = toolkits.reshape_variable(test_X_indicating_mask)
            test_X_ori_variable = toolkits.reshape_variable(test_X_ori_variable)

        elif(original == False):
            test_X_indicating_mask_variable = toolkits.pre_reshape(test_X_indicating_mask)
            test_X_ori_variable = toolkits.pre_reshape(test_X_ori)
            test_X_indicating_mask_variable = toolkits.reshape_variable(test_X_indicating_mask_variable)
            test_X_ori_variable = toolkits.reshape_variable(test_X_ori_variable)

        return test_X_indicating_mask_variable, test_X_ori_variable
    

    def model_imputation(dataset_for_testing, model):
        model_imputation = []
        for value in  dataset_for_testing.values():
            _dict = {'X':value}
            model_results = model.predict(_dict)
            model_imputation.append(model_results["imputation"])
        
        return model_imputation
    
    def calculate_mae(model_imputation, test_X_ori, indicating_mask):
        testing_mae_model_append_subgroups = []
        testing_mae_model_append_variables = []
        for i in range(len(model_imputation)):
            for j in range(len(model_imputation[i])):
                testing_mae_model_append_variables.append(calc_mae(model_imputation[i][j], test_X_ori[i][j], indicating_mask[i][j]))
        testing_mae_model_append_subgroups.append(testing_mae_model_append_variables)
        testing_mae_model_append_variables = [] 

        return testing_mae_model_append_subgroups
    
    #Mae per model
    def show_mae(testing_mae_model, subgroups, variables):

        for i in range(len(subgroups)): 
                print(subgroups[i]) 
                print("-------------")
                for j in range(len(variables)):
                    print(variables[j], ":" ,testing_mae_model[i][j])

    
    #Create table per model
    def create_table(testing_mae_model, subgroups, variables):
        
        df_model_mae = pd.DataFrame(variables)

        for i in range(len(subgroups)):
                df_model_mae[subgroups[i]] = testing_mae_model[i]


        return df_model_mae
        
