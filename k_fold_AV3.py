import numpy as np
import time

from knn import KNN
from linear_regression_m import MRegression
from mlp import MLPRegressor


# Cria os folds manualmente, sem utilizar bibliotecas prontas.
# Na regressao, mantém a mesma divisao aleatoria utilizada no k_fold.py.
def _criar_folds(n_amostras, k_folds=5, random_state=42):
    if k_folds < 2 or k_folds > n_amostras:
        raise ValueError(
            "k_folds deve ser pelo menos 2 e nao pode superar o numero de amostras."
        )

    gerador = np.random.default_rng(random_state)
    indices = np.arange(n_amostras)
    gerador.shuffle(indices)

    folds = [
        lista.tolist()
        for lista in np.array_split(indices, k_folds)
    ]

    return [np.asarray(fold, dtype=int) for fold in folds]


# Padronizacao Z-score.
# Calcula media e desvio apenas com os dados de treinamento.
def _padronizar(X_train, X_test):
    media = np.mean(X_train, axis=0)
    desvio = np.std(X_train, axis=0)

    # Evita divisao por zero.
    desvio[desvio == 0] = 1

    X_train = (X_train - media) / desvio
    X_test = (X_test - media) / desvio

    return X_train, X_test


# Treina o algoritmo escolhido.
# Mantem a compatibilidade com os regressores existentes no projeto.
def _treinar_modelo(
    modelo,
    X_train,
    y_train,
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
        regressor = KNN(
            k=k_vizinhos,
            task="regression",
            distance=distancia
        )
        regressor.fit(X_train, y_train)

    elif modelo == "linear_regression":
        regressor = MRegression(X_train, y_train)
        regressor.fit()

    elif modelo == "mlp":
        regressor = MLPRegressor(
            hidden_layers=hidden_layers,
            activation=activation,
            learning_rate=learning_rate,
            epochs=epochs,
            random_state=random_state
        )
        regressor.fit(X_train, y_train)

    else:
        raise ValueError(
            "Modelo de regressao invalido. Use 'knn', 'linear_regression' ou 'mlp'."
        )

    tempo_treino = time.perf_counter() - inicio

    return regressor, tempo_treino


# Calcula as metricas de regressao para um conjunto de dados.
def _calcular_metricas_regressao(y_real, y_previsto, n_atributos):
    y_real = np.asarray(y_real, dtype=np.float64).reshape(-1)
    y_previsto = np.asarray(y_previsto, dtype=np.float64).reshape(-1)

    erro = y_real - y_previsto

    mse = np.mean(erro ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(erro))

    denominador = np.sum((y_real - np.mean(y_real)) ** 2)

    r2 = (
        1 - np.sum(erro ** 2) / denominador
        if denominador != 0
        else 0.0
    )

    n = len(y_real)

    if n > n_atributos + 1:
        r2_ajustado = (
            1 - (1 - r2) * (n - 1) / (n - n_atributos - 1)
        )
    else:
        r2_ajustado = np.nan

    return {
        "r2_score": float(r2),
        "r2_score_ajus": float(r2_ajustado),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
    }


# Calcula medias e desvios padrao das metricas dos folds.
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
# REGRESSAO - K-FOLD AV3
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
    """Executa K-Fold de regressao e calcula metricas de treino e teste.

    A interface e as chaves originais de retorno ("media", "desvio_padrao"
    e "resultados_folds") sao mantidas para que scripts anteriores da AV3
    continuem funcionando. As metricas adicionais de treinamento ficam em
    "media_treino", "desvio_padrao_treino" e "resultados_folds_treino".
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X deve ser uma matriz bidimensional.")

    if len(X) != len(y):
        raise ValueError("X e y devem possuir a mesma quantidade de amostras.")

    folds = _criar_folds(
        len(X),
        k_folds,
        random_state=random_state
    )

    # Resultados do conjunto de teste: mantem o formato original da AV2.
    resultados_folds = []

    # Resultados adicionais do conjunto de treinamento.
    resultados_folds_treino = []

    for i in range(k_folds):
        # Separa os indices de treinamento e teste.
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

        # Aplica Z-score usando somente os dados de treinamento.
        if padronizar:
            X_train, X_test = _padronizar(
                X_train,
                X_test
            )

        # Treina o modelo. O tempo contabilizado e somente o treinamento.
        regressor, tempo_treino = _treinar_modelo(
            modelo,
            X_train,
            y_train,
            distancia,
            k_vizinhos,
            learning_rate=learning_rate,
            epochs=epochs,
            hidden_layers=hidden_layers,
            activation=activation,
            random_state=random_state
        )

        # TESTE: mantem a mesma ordem e a mesma medicao do k_fold.py original.
        inicio = time.perf_counter()

        y_pre = np.asarray(
            regressor.predict(X_test),
            dtype=np.float64
        ).reshape(-1)

        tempo_teste = time.perf_counter() - inicio

        metricas_teste = _calcular_metricas_regressao(
            y_test,
            y_pre,
            X_train.shape[1]
        )

        # Mantem as chaves originais, inclusive os tempos, para compatibilidade.
        resultados_folds.append({
            "r2_score": metricas_teste["r2_score"],
            "r2_score_ajus": metricas_teste["r2_score_ajus"],
            "mae": metricas_teste["mae"],
            "tempo_treino": float(tempo_treino),
            "tempo_teste": float(tempo_teste),
            "mse": metricas_teste["mse"],
            "rmse": metricas_teste["rmse"],
        })

        # TREINAMENTO: previsoes nas mesmas amostras usadas para ajustar o modelo.
        # Esta avaliacao adicional nao altera o modelo nem suas previsoes de teste.
        y_pre_treino = np.asarray(
            regressor.predict(X_train),
            dtype=np.float64
        ).reshape(-1)

        metricas_treino = _calcular_metricas_regressao(
            y_train,
            y_pre_treino,
            X_train.shape[1]
        )

        resultados_folds_treino.append(metricas_treino)

    # Estatisticas de teste: mesmas metricas e formato de retorno da funcao original.
    media, desvio_padrao = _resumir_resultados(
        resultados_folds
    )

    # Estatisticas adicionais de treinamento.
    media_treino, desvio_padrao_treino = _resumir_resultados(
        resultados_folds_treino
    )

    return {
        "modelo": modelo,
        "tarefa": "regression",
        "k_folds": k_folds,
        "distancia": distancia if modelo == "knn" else None,
        "resultados_folds": resultados_folds,
        "media": media,
        "desvio_padrao": desvio_padrao,
        "resultados_folds_treino": resultados_folds_treino,
        "media_treino": media_treino,
        "desvio_padrao_treino": desvio_padrao_treino,
    }
