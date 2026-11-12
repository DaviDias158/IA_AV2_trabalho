
import numpy as np

from k_fold import k_fold_regressao


# Leitura dos dados
dados = np.loadtxt(
    "cpu_small_227.csv",
    delimiter=",",
    skiprows=1
)

X = dados[:, 1:-1]
y = dados[:, -1]


# Modelos utilizados na comparação
modelos = [
    ("KNN Euclidiana", {
        "modelo": "knn",
        "distancia": "euclidean",
        "k_vizinhos": 5
    }),
    ("KNN Manhattan", {
        "modelo": "knn",
        "distancia": "manhattan",
        "k_vizinhos": 5
    }),
    ("Regressao Linear", {
        "modelo": "linear_regression"
    }),
    ("MLP (8, ReLU)", {
        "modelo": "mlp",
        "hidden_layers": (8,),
        "activation": "relu",
        "learning_rate": 0.1,
        "epochs": 300
    })
]


resultados = []

print("ANÁLISE DO IMPACTO DO Z-SCORE - REGRESSÃO")


for nome, parametros in modelos:
    for padronizar in [False, True]:

        normalizacao = "Com Z-score" if padronizar else "Sem Z-score"

        print(f"\nExecutando {nome} - {normalizacao}")

        try:
            # Identifica possíveis problemas numéricos
            with np.errstate(
                over="raise",
                invalid="raise",
                divide="raise"
            ):
                resultado = k_fold_regressao(
                    X,
                    y,
                    k_folds=5,
                    padronizar=padronizar,
                    random_state=42,
                    **parametros
                )

            media = resultado["media"]

            # Verifica se as métricas são valores válidos
            metricas = ["mse", "rmse", "mae", "r2_score"]

            if not all(np.isfinite(media[m]) for m in metricas):
                print("Resultado inválido: instabilidade numérica.")
                continue

            resultados.append((nome, normalizacao, resultado))

            print(f"RMSE: {media['rmse']:.4f}")
            print(f"MAE: {media['mae']:.4f}")
            print(f"R2: {media['r2_score']:.4f}")

        except (FloatingPointError, OverflowError,
                np.linalg.LinAlgError) as erro:

            print("Treinamento apresentou instabilidade numérica.")
            print("Detalhe:", erro)


# Tabela comparativa
print("\n" + "=" * 105)
print("COMPARAÇÃO COM E SEM Z-SCORE")
print("=" * 105)

print(
    f"{'Modelo':<20} | {'Normalização':<12} | "
    f"{'RMSE':<10} | {'MAE':<10} | "
    f"{'R2':<10} | {'Treino (s)':<10}"
)

print("-" * 105)

for nome, normalizacao, resultado in resultados:
    media = resultado["media"]

    print(
        f"{nome:<20} | "
        f"{normalizacao:<12} | "
        f"{media['rmse']:<10.4f} | "
        f"{media['mae']:<10.4f} | "
        f"{media['r2_score']:<10.4f} | "
        f"{media['tempo_treino']:<10.4f}"
    )
