
import numpy as np
import time

from knn import KNN
from nb_gaussiano import NaiveBayes
from linear_regression_m import MRegression
from perceptron_multiclasse import PerceptronMulticlasse
from mlp import MLPRegressor

from matrix_confusion import (
    accuracy,
    precision,
    recall,
    f1_score,
    specificity,
    confusion_matrix
)


# Cria os folds manualmente, sem utilizar scikit-learn.
# Na classificação, mantém a proporção das classes entre os folds.
def _criar_folds(n_amostras, k_folds=5, y=None, random_state=42):
    if k_folds < 2 or k_folds > n_amostras:
        raise ValueError(
            "k_folds deve ser pelo menos 2 e não pode superar o número de amostras."
        )

    gerador = np.random.default_rng(random_state)
    folds = [[] for _ in range(k_folds)]

    if y is not None:
        # Classificação: folds estratificados
        y = np.asarray(y)

        for classe in np.unique(y):
            indices_classe = np.where(y == classe)[0]
            gerador.shuffle(indices_classe)

            for i, indice in enumerate(indices_classe):
                folds[i % k_folds].append(indice)

    else:
        # Regressão: divisão comum em k partes
        indices = np.arange(n_amostras)
        gerador.shuffle(indices)

        folds = [
            lista.tolist()
            for lista in np.array_split(indices, k_folds)
        ]

    return [np.asarray(fold, dtype=int) for fold in folds]


# Padronização Z-score.
# Calcula média e desvio apenas com os dados de treinamento.
def _padronizar(X_train, X_test):
    media = np.mean(X_train, axis=0)
    desvio = np.std(X_train, axis=0)

    # Evita divisão por zero
    desvio[desvio == 0] = 1

    X_train = (X_train - media) / desvio
    X_test = (X_test - media) / desvio

    return X_train, X_test


# Treina o algoritmo escolhido.
# Inclui os modelos da AV2 e o novo Perceptron Multiclasse.
def _treinar_modelo(
    modelo,
    X_train,
    y_train,
    tarefa,
    distancia,
    k_vizinhos,
    learning_rate=0.01,
    epochs=100,
    hidden_layers=(16,),
    activation="tanh",
    random_state=42
):
    inicio = time.perf_counter()

    if modelo == "knn":
        classificador = KNN(
            k=k_vizinhos,
            task=tarefa,
            distance=distancia
        )
        classificador.fit(X_train, y_train)

    elif modelo == "naive_bayes" and tarefa == "classification":
        classificador = NaiveBayes()
        classificador.fit(X_train, y_train)

    elif modelo == "linear_regression" and tarefa == "regression":
        classificador = MRegression(X_train, y_train)
        classificador.fit()

    elif modelo == "perceptron" and tarefa == "classification":
        classificador = PerceptronMulticlasse(
            learning_rate=learning_rate,
            epochs=epochs
        )
        classificador.fit(X_train, y_train)


    elif modelo == "mlp" and tarefa == "regression":
        classificador = MLPRegressor(
            hidden_layers=hidden_layers,
            activation=activation,
            learning_rate=learning_rate,
            epochs=epochs,
            random_state=random_state
        )

        classificador.fit(X_train, y_train)

    else:
        raise ValueError(
            "Modelo incompatível com a tarefa ou nome de modelo inválido."
        )

    tempo_treino = time.perf_counter() - inicio

    return classificador, tempo_treino


# Calcula médias e desvios padrão das métricas dos folds.
def _resumir_resultados(resultados_folds):
    metricas = resultados_folds[0].keys()

    media = {}
    desvio_padrao = {}

    for metrica in metricas:
        valores = np.array(
            [resultado[metrica] for resultado in resultados_folds],
            dtype=float
        )

        media[metrica] = float(np.mean(valores))
        desvio_padrao[metrica] = float(np.std(valores, ddof=1))

    return media, desvio_padrao


# ==========================================================
# CLASSIFICAÇÃO - K-FOLD
# ==========================================================

def k_fold_classificacao(
    X,
    y,
    modelo="knn",
    k_folds=5,
    distancia="euclidean",
    k_vizinhos=5,
    padronizar=True,
    random_state=42,
    learning_rate=0.01,
    epochs=100
):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    folds = _criar_folds(
        len(X),
        k_folds,
        y=y,
        random_state=random_state
    )

    resultados_folds = []

    # Armazena os resultados para a matriz de confusão geral
    reais_por_fold = []
    previstos_por_fold = []

    for i in range(k_folds):

        # Separa os índices de treinamento e teste
        indices_teste = folds[i]

        indices_treino = np.concatenate([
            folds[j]
            for j in range(k_folds)
            if j != i
        ])

        X_train = X[indices_treino]
        X_test = X[indices_teste]

        y_train = y[indices_treino]
        y_test = y[indices_teste]

        # Aplica Z-score, quando solicitado
        if padronizar:
            X_train, X_test = _padronizar(
                X_train,
                X_test
            )

        # Treinamento
        classificador, tempo_treino = _treinar_modelo(
            modelo,
            X_train,
            y_train,
            "classification",
            distancia,
            k_vizinhos,
            learning_rate=learning_rate,
            epochs=epochs
        )

        # Teste
        inicio = time.perf_counter()

        y_pre = np.asarray(classificador.predict(X_test))

        tempo_teste = time.perf_counter() - inicio

        # Guarda os valores reais e previstos
        reais_por_fold.append(y_test)
        previstos_por_fold.append(y_pre)

        # Calcula as métricas de cada fold
        resultados_folds.append({
            "acuracia": float(accuracy(y_test, y_pre)),
            "precisao": float(precision(y_test, y_pre)),
            "recall": float(recall(y_test, y_pre)),
            "especificidade": float(specificity(y_test, y_pre)),
            "f1_score": float(f1_score(y_test, y_pre)),
            "tempo_treino": float(tempo_treino),
            "tempo_teste": float(tempo_teste),
        })

    # Calcula médias e desvios padrão
    media, desvio_padrao = _resumir_resultados(
        resultados_folds
    )

    # Junta as previsões dos cinco folds
    y_reais = np.concatenate(reais_por_fold)
    y_previstos = np.concatenate(previstos_por_fold)

    # Calcula a matriz de confusão geral
    matriz_total, classes = confusion_matrix(
        y_reais,
        y_previstos
    )

    return {
        "modelo": modelo,
        "tarefa": "classification",
        "k_folds": k_folds,
        "distancia": distancia if modelo == "knn" else None,
        "resultados_folds": resultados_folds,
        "media": media,
        "desvio_padrao": desvio_padrao,
        "matriz_confusao": matriz_total,
        "classes": classes,
    }


# ==========================================================
# REGRESSÃO - K-FOLD
# ==========================================================


def k_fold_regressao(
    X,
    y,
    modelo="knn",
    k_folds=5,
    distancia="euclidean",
    k_vizinhos=5,
    padronizar=True,
    random_state=42,
    hidden_layers=(16,),
    activation="tanh",
    learning_rate=0.01,
    epochs=100
):

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    folds = _criar_folds(
        len(X),
        k_folds,
        random_state=random_state
    )

    resultados_folds = []

    for i in range(k_folds):

        # Separa os índices de treinamento e teste
        indices_teste = folds[i]

        indices_treino = np.concatenate([
            folds[j]
            for j in range(k_folds)
            if j != i
        ])

        X_train = X[indices_treino]
        X_test = X[indices_teste]

        y_train = y[indices_treino]
        y_test = y[indices_teste]

        # Padronização Z-score
        if padronizar:
            X_train, X_test = _padronizar(
                X_train,
                X_test
            )

        # Treinamento
       
        regressor, tempo_treino = _treinar_modelo(
            modelo,
            X_train,
            y_train,
            "regression",
            distancia,
            k_vizinhos,
            learning_rate=learning_rate,
            epochs=epochs,
            hidden_layers=hidden_layers,
            activation=activation,
            random_state=random_state
        )

        # Teste
        inicio = time.perf_counter()

        y_pre = np.asarray(
            regressor.predict(X_test),
            dtype=np.float64
        ).reshape(-1)

        tempo_teste = time.perf_counter() - inicio

        # Cálculo das métricas de regressão
        erro = y_test - y_pre

        mse = np.mean(erro ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(erro))

        denominador = np.sum(
            (y_test - np.mean(y_test)) ** 2
        )

        r2 = (
            1 - np.sum(erro ** 2) / denominador
            if denominador != 0
            else 0.0
        )

        n = len(y_test)
        p = X_train.shape[1]

        if n > p + 1:
            r2_ajustado = (
                1 - (1 - r2) * (n - 1) / (n - p - 1)
            )
        else:
            r2_ajustado = np.nan

        resultados_folds.append({
            "r2_score": float(r2),
            "r2_score_ajus": float(r2_ajustado),
            "mae": float(mae),
            "tempo_treino": float(tempo_treino),
            "tempo_teste": float(tempo_teste),
            "mse": float(mse),
            "rmse": float(rmse),
        })

    media, desvio_padrao = _resumir_resultados(
        resultados_folds
    )

    return {
        "modelo": modelo,
        "tarefa": "regression",
        "k_folds": k_folds,
        "distancia": distancia if modelo == "knn" else None,
        "resultados_folds": resultados_folds,
        "media": media,
        "desvio_padrao": desvio_padrao,
    }
