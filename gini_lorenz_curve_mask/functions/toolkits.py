import matplotlib.pyplot as plt
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
    
    def lorenz_curve(X):
        X = np.sort(X)  # ordenar os dados
        X_lorenz = X.cumsum() / X.sum()
        X_lorenz = np.insert(X_lorenz, 0, 0)  # inserir 0 no início
        x_vals = np.linspace(0, 1, len(X_lorenz))  # eixo x normalizado

        fig, ax = plt.subplots(figsize=[6,6])
        
        # Curva de Lorenz com linha contínua
        ax.plot(x_vals, X_lorenz, color='darkgreen', linewidth=2)
        
        # Linha de igualdade
        ax.plot([0, 1], [0, 1], color='black', linestyle='--')
        
        ax.set_title("Curva de Lorenz")
        ax.set_xlabel("População acumulada (fração)")
        ax.set_ylabel("Renda acumulada (fração)")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def lorenz_curve_5(X1, X2, X3, X4, X5, labels=None, colors=None):
        arrays = [X1, X2, X3, X4, X5]
        
        if labels is None:
            labels = ["SAITS", "BRITS", "US-GAN", "GP-VAE", "MRNN"]
        if colors is None:
            colors = ['blue', 'green', 'orange', 'purple', 'red']
        
        fig, ax = plt.subplots(figsize=[6,6])

        for i, X in enumerate(arrays):
            X = np.sort(X)
            X_lorenz = X.cumsum() / X.sum()
            X_lorenz = np.insert(X_lorenz, 0, 0)
            x_vals = np.linspace(0, 1, len(X_lorenz))

            ax.plot(x_vals, X_lorenz, linewidth=2, label=labels[i], color=colors[i])

        # Linha de igualdade
        ax.plot([0, 1], [0, 1], color='black', linestyle='--', label='perfect equality')

        ax.set_title("Lorenz Curve - 5 Models")
        ax.set_xlabel("Patients")
        ax.set_ylabel("Absolute error (AE)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()