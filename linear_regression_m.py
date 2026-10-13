import numpy as np

class MRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.beta = None
        self.N = X.shape[0]

    def fit(self):
        X = np.column_stack((np.ones(self.N), self.X))

        self.beta = np.linalg.pinv(X) @ self.y

    def predict(self, X_new):
        N = X_new.shape[0]
        X_new = np.column_stack((np.ones(N), X_new))
        return X_new @ self.beta

    def r2_score(self, y_true, y_preditivo):
        numerador = np.sum((y_true - y_preditivo) ** 2)
        denominador = np.sum((y_true - np.mean(y_true)) ** 2)

        r2_score = 1 - (numerador / denominador)

        return r2_score

    def r2_score_ajus(self, r2, n, p):
        numerador = (1 - r2) * (n - 1)
        denominador = (n - p - 1)

        r2_score_ajus = 1 - (numerador / denominador)

        return r2_score_ajus

    def rmse(self, y_true, y_pred):
        erro = (y_true - y_pred) ** 2

        media = np.mean(erro)

        return np.sqrt(media)

    def mse(self, y_true, y_pred):
        erro = (y_true - y_pred) ** 2

        return np.mean(erro)

    def mae(self, y_true, y_pred):
        erro = np.abs(y_true - y_pred)

        return np.mean(erro)

    def rss(self, y_true, y_pred):
        erro = y_true - y_pred

        return np.sum(erro ** 2)
