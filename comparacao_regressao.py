
import numpy as np

from k_fold import k_fold_regressao


# Leitura da base de regressão
dados = np.loadtxt(
    "cpu_small_227.csv",
    delimiter=",",
    skiprows=1
)

X = dados[:, 1:-1]
y = dados[:, -1]


# Modelos que serão comparados
modelos = [
    (
        "KNN Euclidiana",
        {
            "modelo": "knn",
            "distancia": "euclidean",
            "k_vizinhos": 5
        }
    ),
    (
        "KNN Manhattan",
        {
            "modelo": "knn",
            "distancia": "manhattan",
            "k_vizinhos": 5
        }
    ),
    (
        "Regressao Linear",
        {
            "modelo": "linear_regression"
        }
    ),
    (
        "MLP (8, ReLU)",
        {
            "modelo": "mlp",
            "hidden_layers": (8,),
            "activation": "relu",
            "learning_rate": 0.1,
            "epochs": 300
        }
    )
]


resultados = []

print("COMPARAÇÃO DOS MODELOS DE REGRESSÃO")
print("Amostras:", len(X))
print("Atributos:", X.shape[1])
print("Validação cruzada: 5 folds")
print("Normalização: Z-score")


# Executa os modelos
for nome, parametros in modelos:

    print(f"\nExecutando: {nome}")

    resultado = k_fold_regressao(
        X,
        y,
        k_folds=5,
        padronizar=True,
        random_state=42,
        **parametros
    )

    resultados.append((nome, resultado))

    print("Concluído.")


# Exibe a tabela comparativa
print("\n" + "=" * 125)
print("RESULTADOS - MÉDIAS DOS 5 FOLDS")
print("=" * 125)

print(
    f"{'Modelo':<20} | "
    f"{'MSE':<9} | "
    f"{'RMSE':<9} | "
    f"{'MAE':<9} | "
    f"{'R2':<8} | "
    f"{'R2 Ajust.':<10} | "
    f"{'Treino (s)':<10} | "
    f"{'Teste (s)':<10}"
)

print("-" * 125)

for nome, resultado in resultados:

    media = resultado["media"]

    print(
        f"{nome:<20} | "
        f"{media['mse']:<9.4f} | "
        f"{media['rmse']:<9.4f} | "
        f"{media['mae']:<9.4f} | "
        f"{media['r2_score']:<8.4f} | "
        f"{media['r2_score_ajus']:<10.4f} | "
        f"{media['tempo_treino']:<10.4f} | "
        f"{media['tempo_teste']:<10.4f}"
    )


# Seleciona o modelo com menor RMSE médio
melhor = min(
    resultados,
    key=lambda item: item[1]["media"]["rmse"]
)

print("\n" + "=" * 60)
print("MELHOR MODELO PELO RMSE")
print("=" * 60)

print("Modelo:", melhor[0])

print("\nMétricas - média ± desvio padrão")

for metrica, valor in melhor[1]["media"].items():

    desvio = melhor[1]["desvio_padrao"][metrica]

    print(f"{metrica}: {valor:.4f} ± {desvio:.4f}")
