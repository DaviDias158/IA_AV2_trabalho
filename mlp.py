
import numpy as np


class MLPRegressor:
    def __init__(
        self,
        hidden_layers=(16,),
        activation="tanh",
        learning_rate=0.01,
        epochs=500,
        random_state=42
    ):
        if len(hidden_layers) == 0 or any(n <= 0 for n in hidden_layers):
            raise ValueError("Informe pelo menos uma camada oculta válida.")

        if activation not in ("tanh", "relu"):
            raise ValueError("A ativação deve ser 'tanh' ou 'relu'.")

        if learning_rate <= 0 or epochs < 1:
            raise ValueError("Taxa e épocas devem ser positivas.")

        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        self.weights = []
        self.biases = []
        self.loss_history = []

    # Aplica a função de ativação nas camadas ocultas
    def _ativar(self, z):
        if self.activation == "tanh":
            return np.tanh(z)

        return np.maximum(0, z)

    # Derivada da ativação, utilizada na retropropagação
    def _derivada_ativacao(self, z):
        if self.activation == "tanh":
            return 1 - np.tanh(z) ** 2

        return (z > 0).astype(float)

    # Inicializa os pesos e biases da rede
    def _inicializar(self, n_features):
        gerador = np.random.default_rng(self.random_state)

        tamanhos = [n_features, *self.hidden_layers, 1]

        self.weights = []
        self.biases = []

        for entrada, saida in zip(tamanhos[:-1], tamanhos[1:]):
            limite = np.sqrt(6 / (entrada + saida))

            pesos = gerador.uniform(
                -limite,
                limite,
                size=(entrada, saida)
            )

            bias = np.zeros((1, saida))

            self.weights.append(pesos)
            self.biases.append(bias)

    # Treinamento da rede neural
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if X.ndim != 2 or len(X) != len(y) or len(X) == 0:
            raise ValueError("X e y devem possuir amostras compatíveis.")

        # Padroniza a variável alvo usando apenas o treino
        self.media_y = float(np.mean(y))
        self.desvio_y = float(np.std(y))

        if self.desvio_y == 0:
            self.desvio_y = 1.0

        y_padronizado = (y - self.media_y) / self.desvio_y

        self._inicializar(X.shape[1])
        self.loss_history = []

        n_amostras = X.shape[0]

        for _ in range(self.epochs):

            # FORWARD PROPAGATION
            ativacoes = [X]
            somas_ocultas = []
            a = X

            # Percorre as camadas ocultas
            for W, b in zip(self.weights[:-1], self.biases[:-1]):
                z = a @ W + b
                somas_ocultas.append(z)

                a = self._ativar(z)
                ativacoes.append(a)

            # Camada de saída com ativação linear
            saida = a @ self.weights[-1] + self.biases[-1]
            ativacoes.append(saida)

            # Calcula o erro
            erro = saida - y_padronizado

            mse = np.mean(erro ** 2)
            self.loss_history.append(float(mse))

            # BACKPROPAGATION
            delta = 2 * erro / n_amostras

            grad_pesos = [None] * len(self.weights)
            grad_bias = [None] * len(self.biases)

            # Percorre as camadas de trás para frente
            for camada in range(len(self.weights) - 1, -1, -1):

                grad_pesos[camada] = (
                    ativacoes[camada].T @ delta
                )

                grad_bias[camada] = np.sum(
                    delta,
                    axis=0,
                    keepdims=True
                )

                if camada > 0:
                    delta = (
                        delta @ self.weights[camada].T
                    ) * self._derivada_ativacao(
                        somas_ocultas[camada - 1]
                    )

            # Atualiza pesos e biases
            for camada in range(len(self.weights)):
                self.weights[camada] -= (
                    self.learning_rate * grad_pesos[camada]
                )

                self.biases[camada] -= (
                    self.learning_rate * grad_bias[camada]
                )

    # Realiza previsões
    def predict(self, X):
        if not self.weights:
            raise ValueError("Treine o modelo antes de prever.")

        X = np.asarray(X, dtype=float)
        a = X

        # Propagação pelas camadas ocultas
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            a = self._ativar(a @ W + b)

        # Saída linear
        saida = a @ self.weights[-1] + self.biases[-1]

        # Retorna à escala original da variável alvo
        previsoes = saida * self.desvio_y + self.media_y

        return previsoes.reshape(-1)
