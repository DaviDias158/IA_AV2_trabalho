import numpy as np
from perceptron import Perceptron
import perceptron
print("Arquivo utilizado:", perceptron.__file__)

class PerceptronMulticlasse:
    def __init__(self, learning_rate=100, epochs=100):
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

            print(
                f"\nClasse: {classe} | "
                f"Learning rate: {self.learning_rate}"
            )

            print("Atualizações:", modelo.n_updates)

            print(
                "Pesos / Learning rate:",
                np.round(
                    modelo.weights / self.learning_rate,
                    6
                )
            )

            print(
                "Bias / Learning rate:",
                round(
                    modelo.bias / self.learning_rate,
                    6
                )
            )

            self.modelos_.append(modelo)

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        respostas_binarias = []
        pontuacoes = []

        for modelo in self.modelos_:
            respostas_binarias.append(modelo.predict(X))
            pontuacoes.append(modelo.decision_function(X))

        respostas_binarias = np.column_stack(respostas_binarias)
        pontuacoes = np.column_stack(pontuacoes)

        # Decisão inicial: maior pontuação
        indices = np.argmax(pontuacoes, axis=1)

        # Quando somente um classificador responde 1,
        # seleciona diretamente a classe correspondente.
        for i in range(len(X)):

            positivos = np.where(respostas_binarias[i] == 1)[0]

            if len(positivos) == 1:
                indices[i] = positivos[0]

        return self.classes_[indices]
