
import time
import numpy as np

from mlp import MLPRegressor
from k_fold import _criar_folds, _padronizar


# Leitura da base de regressão
dados = np.loadtxt(
    "cpu_small_227.csv",
    delimiter=",",
    skiprows=1
)

X = dados[:, 1:-1]
y = dados[:, -1]


# Calcula as métricas de regressão
def calcular_metricas(y_real, y_previsto):
    erro = y_real - y_previsto

    mse = np.mean(erro ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(erro))

    denominador = np.sum(
        (y_real - np.mean(y_real)) ** 2
    )

    r2 = (
        1 - np.sum(erro ** 2) / denominador
        if denominador != 0
        else 0.0
    )

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }


# Mesmos folds utilizados nos experimentos anteriores
folds = _criar_folds(
    len(X),
    k_folds=5,
    random_state=42
)

resultados = []

print("ANÁLISE DE GENERALIZAÇÃO - MLP")
print("Amostras:", len(X))
print("Atributos:", X.shape[1])


for i in range(5):

    indices_teste = folds[i]

    indices_treino = np.concatenate([
        folds[j]
        for j in range(5)
        if j != i
    ])

    X_train = X[indices_treino]
    X_test = X[indices_teste]

    y_train = y[indices_treino]
    y_test = y[indices_teste]

    # Normalização somente com parâmetros do treino
    X_train, X_test = _padronizar(
        X_train,
        X_test
    )

    modelo = MLPRegressor(
        hidden_layers=(8,),
        activation="relu",
        learning_rate=0.1,
        epochs=300,
        random_state=42
    )

    # Treinamento
    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    tempo_treino = time.perf_counter() - inicio

    # Previsões no conjunto de treinamento
    y_pred_train = modelo.predict(X_train)

    # Previsões no conjunto de teste
    inicio = time.perf_counter()
    y_pred_test = modelo.predict(X_test)
    tempo_teste = time.perf_counter() - inicio

    metricas_treino = calcular_metricas(
        y_train,
        y_pred_train
    )

    metricas_teste = calcular_metricas(
        y_test,
        y_pred_test
    )

    resultados.append({
        "treino": metricas_treino,
        "teste": metricas_teste,
        "tempo_treino": tempo_treino,
        "tempo_teste": tempo_teste
    })

    print(f"\nFOLD {i + 1}")

    print(
        f"Treino: RMSE={metricas_treino['rmse']:.4f} | "
        f"R2={metricas_treino['r2']:.4f}"
    )

    print(
        f"Teste:  RMSE={metricas_teste['rmse']:.4f} | "
        f"R2={metricas_teste['r2']:.4f}"
    )


print("\n" + "=" * 60)
print("RESULTADOS FINAIS - MÉDIA DOS 5 FOLDS")
print("=" * 60)


for metrica in ["rmse", "mae", "r2"]:

    treino = np.array([
        resultado["treino"][metrica]
        for resultado in resultados
    ])

    teste = np.array([
        resultado["teste"][metrica]
        for resultado in resultados
    ])

    print(f"\n{metrica.upper()}")

    print(
        f"Treino: {np.mean(treino):.4f} "
        f"± {np.std(treino, ddof=1):.4f}"
    )

    print(
        f"Teste:  {np.mean(teste):.4f} "
        f"± {np.std(teste, ddof=1):.4f}"
    )


rmse_treino = np.mean([
    r["treino"]["rmse"] for r in resultados
])

rmse_teste = np.mean([
    r["teste"]["rmse"] for r in resultados
])

r2_treino = np.mean([
    r["treino"]["r2"] for r in resultados
])

r2_teste = np.mean([
    r["teste"]["r2"] for r in resultados
])


print("\n" + "=" * 60)
print("DIFERENÇA ENTRE TREINAMENTO E TESTE")
print("=" * 60)

print(f"Diferença RMSE: {rmse_teste - rmse_treino:.4f}")
print(f"Diferença R2: {r2_treino - r2_teste:.4f}")


tempo_medio_treino = np.mean([
    r["tempo_treino"] for r in resultados
])

tempo_medio_teste = np.mean([
    r["tempo_teste"] for r in resultados
])

print("\nTEMPO MÉDIO POR FOLD")
print(f"Treinamento: {tempo_medio_treino:.4f} s")
print(f"Teste: {tempo_medio_teste:.4f} s")
