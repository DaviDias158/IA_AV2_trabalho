
import numpy as np
from perceptron import Perceptron


class PerceptronMulticlasse:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.classes_ = None
        self.modelos_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Identifica todas as classes presentes
        self.classes_ = np.unique(y)
        self.modelos_ = []

        # Treina um Perceptron para cada classe
        for classe in self.classes_:

            # A classe atual recebe 1 e as demais recebem 0
            y_binario = (y == classe).astype(int)

            modelo = Perceptron(
                learning_rate=self.learning_rate,
                epochs=self.epochs
            )

            modelo.fit(X, y_binario)
            self.modelos_.append(modelo)

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        # Calcula a soma ponderada de cada Perceptron
        pontuacoes = []

        for modelo in self.modelos_:
            pontuacoes.append(modelo.decision_function(X))

        pontuacoes = np.column_stack(pontuacoes)

        # Seleciona a classe com maior pontuacao
        indices = np.argmax(pontuacoes, axis=1)

        return self.classes_[indices]
