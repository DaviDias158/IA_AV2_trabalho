
import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Identifica a quantidade de amostras e atributos
        n_samples, n_features = X.shape

        # Inicializa pesos e bias com zero
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Processo de treinamento
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):

                # Calcula a soma ponderada
                linear_output = np.dot(x_i, self.weights) + self.bias

                # Aplica a função de ativação
                y_predicted = self._step_function(linear_output)

                # Ajusta os pesos quando ocorre um erro
                if y[idx] != y_predicted:
                    update = self.learning_rate * (y[idx] - y_predicted)

                    self.weights += update * x_i
                    self.bias += update

    def predict(self, X):
        # Calcula a saída para as amostras
        linear_output = np.dot(X, self.weights) + self.bias

        return self._step_function(linear_output)

    def _step_function(self, x):
        return np.where(x >= 0, 1, 0)


    
    def decision_function(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return linear_output

