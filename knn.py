import numpy as np


# KNN adaptado para classificação e regressão.
class KNN:
    def __init__(self, k=5, task="classification", distance="euclidean"):
        self.k = k
        self.task = task
        self.distance = distance  # ALTERAÇÃO: permite escolher euclidiana ou Manhattan.

    def fit(self, X_train, y_train):
        # metodo do treinamento da maquina
        self.X_train = np.asarray(X_train) #ALTERAÇÃO
        self.y_train = np.asarray(y_train) #ALTERAÇÃO

    def calculate_distances(self, x1, x2):
        # ALTERAÇÃO: mantém a distância euclidiana da professora como padrão
        # e acrescenta a distância Manhattan exigida no trabalho.
        if self.distance == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance == "manhattan":
            return np.sum(np.abs(x1 - x2))
        else:
            # ALTERAÇÃO: sinaliza uma opção de distância inválida com um erro.
            raise ValueError("A distância deve ser 'euclidean' ou 'manhattan'.")

    def calculate_predict(self, x): #ALTERAÇÃO, MUITO DIFERENTE DO DA CYNTHIA
            # ALTERAÇÃO: calcula todas as distâncias com operações vetorizadas do NumPy.
            # O resultado é o mesmo do cálculo individual, mas evita milhares de
            # chamadas Python para cada amostra, que deixavam o K-Fold muito lento.
            if self.distance == "euclidean":
                distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            elif self.distance == "manhattan":
                distances = np.sum(np.abs(self.X_train - x), axis=1)
            else:
                raise ValueError("A distância deve ser 'euclidean' ou 'manhattan'.")
    
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
    
            if self.task == "classification":
                unique, counts = np.unique(k_nearest_labels, return_counts=True)
                return unique[np.argmax(counts)]
            elif self.task == "regression":
                return np.mean(k_nearest_labels)
            else:
                raise ValueError("Tarefa sera regressao ou classificacao!")

    def predict(self, X_test): #ALTERAÇÃO, MUITO DIFERENTE DO DA CYNTHIA
            # Mantém a lógica original da professora: prevê uma amostra por vez.
            predictions = [self.calculate_predict(x) for x in X_test]
            return np.asarray(predictions)
