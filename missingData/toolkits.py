import numpy as np
class toolkits:

    def separaring_dataset(dataset):

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
    
    def indicating_mask(dataset_for_testing_ori, dataset_for_testing):
        test_X_indicating_mask = []
        test_X_ori = []
        for i, j in zip(dataset_for_testing_ori.values(), dataset_for_testing.values()):
            test_X_indicating_mask.append(np.isnan(i) ^ np.isnan(j))
            test_X_ori.append(np.nan_to_num(i))

        return test_X_indicating_mask, test_X_ori

    def indicating_mask_variable(dataset_for_testing_ori, dataset_for_testing):
        test_X_indicating_mask_variable = []
        test_X_ori_variable = []
        test_X_indicating_mask, test_X_ori = toolkits.indicating_mask(dataset_for_testing_ori, dataset_for_testing)
        
        for i in range(len(test_X_indicating_mask)):
            test_X_indicating_mask_variable.append(test_X_indicating_mask[i].reshape(37, len(test_X_indicating_mask[i]) * 48))
            test_X_ori_variable.append(test_X_ori[i].reshape(37, len(test_X_ori[i]) * 48))

        return test_X_indicating_mask_variable, test_X_ori_variable