import numpy as np

class toolkits:
    def reshape(dataset):
        for i in range(len(dataset)):
            dataset[i] = dataset[i].reshape(len(dataset[i])* 48 * 37)
        return dataset
    
    def gini(model_ae):
        sorted_ae = model_ae.copy()
        sorted_ae.sort()
        n = model_ae.size
        coef_ = 2./n
        const_ = (n+1.)/n
        weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_ae)])
        return coef_*weighted_sum/(sorted_ae.sum()) - const_
    
    def bootstrap(ae, n_resamples):
        distribution_bootstrap = []

        for i in range(n_resamples):
            indices = np.random.randint(0, len(ae) - 1, size=len(ae))
            resampling_ae = ae[indices]
            gini = toolkits.gini(resampling_ae)
            distribution_bootstrap.append(gini)
        
        return distribution_bootstrap




    
