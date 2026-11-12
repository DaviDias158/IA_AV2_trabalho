
import numpy as np

from k_fold import k_fold_classificacao


# Leitura dos dados de classificação
dados = np.loadtxt(
    "football_player_position_46763.csv",
    delimiter=",",
    skiprows=38
)

X = dados[:, 1:-1]
y = dados[:, -1].astype(int)


# Modelos que serão avaliados
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
    ("Naive Bayes", {
        "modelo": "naive_bayes"
    }),
    ("Perceptron", {
        "modelo": "perceptron",
        "learning_rate": 0.01,
        "epochs": 100
    })
]


resultados = []

print("ANÁLISE DO Z-SCORE - CLASSIFICAÇÃO")
print("Amostras:", len(X))
print("Atributos:", X.shape[1])
print("Classes:", np.unique(y))


# Executa os experimentos
for nome, parametros in modelos:
    for padronizar in [False, True]:

        normalizacao = (
            "Com Z-score" if padronizar
            else "Sem Z-score"
        )

        print(f"\nExecutando {nome} - {normalizacao}")

        resultado = k_fold_classificacao(
            X,
            y,
            k_folds=5,
            padronizar=padronizar,
            random_state=42,
            **parametros
        )

        resultados.append({
            "nome": nome,
            "normalizacao": normalizacao,
            "resultado": resultado
        })

        media = resultado["media"]

        print(f"Acurácia: {media['acuracia']:.4f}")
        print(f"Precisão: {media['precisao']:.4f}")
        print(f"Recall: {media['recall']:.4f}")
        print(f"Especificidade: {media['especificidade']:.4f}")
        print(f"F1-Score: {media['f1_score']:.4f}")


# Tabela comparativa
print("\n" + "=" * 115)
print("COMPARAÇÃO COM E SEM Z-SCORE")
print("=" * 115)

print(
    f"{'Modelo':<16} | "
    f"{'Normalização':<12} | "
    f"{'Acurácia':<9} | "
    f"{'Precisão':<9} | "
    f"{'Recall':<9} | "
    f"{'Especif.':<9} | "
    f"{'F1-Score':<9}"
)

print("-" * 115)

for experimento in resultados:

    media = experimento["resultado"]["media"]

    print(
        f"{experimento['nome']:<16} | "
        f"{experimento['normalizacao']:<12} | "
        f"{media['acuracia']:<9.4f} | "
        f"{media['precisao']:<9.4f} | "
        f"{media['recall']:<9.4f} | "
        f"{media['especificidade']:<9.4f} | "
        f"{media['f1_score']:<9.4f}"
    )


# Compara o efeito da normalização
print("\n" + "=" * 65)
print("IMPACTO DO Z-SCORE")
print("=" * 65)

for nome, parametros in modelos:

    sem = next(
        exp for exp in resultados
        if exp["nome"] == nome
        and exp["normalizacao"] == "Sem Z-score"
    )

    com = next(
        exp for exp in resultados
        if exp["nome"] == nome
        and exp["normalizacao"] == "Com Z-score"
    )

    media_sem = sem["resultado"]["media"]
    media_com = com["resultado"]["media"]

    diferenca_acuracia = (
        media_com["acuracia"] - media_sem["acuracia"]
    )

    diferenca_f1 = (
        media_com["f1_score"] - media_sem["f1_score"]
    )

    print(f"\nModelo: {nome}")

    print(
        f"Diferença na acurácia: "
        f"{diferenca_acuracia * 100:+.2f} pontos percentuais"
    )

    print(
        f"Diferença no F1-Score: "
        f"{diferenca_f1 * 100:+.2f} pontos percentuais"
    )


# Matriz de confusão do Perceptron sem e com Z-score
print("\n" + "=" * 65)
print("MATRIZES DE CONFUSÃO - PERCEPTRON")
print("=" * 65)

for experimento in resultados:

    if experimento["nome"] == "Perceptron":

        resultado = experimento["resultado"]

        print(f"\n{experimento['normalizacao']}")
        print("Classes:", resultado["classes"])
        print(resultado["matriz_confusao"])
