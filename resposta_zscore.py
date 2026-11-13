import numpy as np

from k_fold import k_fold_classificacao
from k_fold_AV3 import k_fold_regressao


# ==========================================================
# LEITURA DOS DATASETS
# ==========================================================

dados_classificacao = np.loadtxt(
    "football_player_position_46763.csv",
    delimiter=",",
    skiprows=38
)

X_classificacao = dados_classificacao[:, 1:-1]
y_classificacao = dados_classificacao[:, -1].astype(int)


dados_regressao = np.loadtxt(
    "cpu_small_227.csv",
    delimiter=",",
    skiprows=1
)

X_regressao = dados_regressao[:, 1:-1]
y_regressao = dados_regressao[:, -1]


# ==========================================================
# CONFIGURACOES DOS EXPERIMENTOS
# ==========================================================

# Mantem os mesmos parametros usados nos scripts de Z-score
# preparados anteriormente pelo grupo.
modelos_classificacao = [
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

modelos_regressao = [
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


# ==========================================================
# FUNCOES AUXILIARES
# ==========================================================

def formatar_resultado(media, desvio):
    return f"{media:.4f} +/- {desvio:.4f}"


def executar_classificacao():
    resultados = []

    print("\n" + "=" * 105)
    print("ANALISE DO IMPACTO DO Z-SCORE - CLASSIFICACAO")
    print("=" * 105)
    print("Quantidade de amostras:", X_classificacao.shape[0])
    print("Quantidade de atributos:", X_classificacao.shape[1])
    print("Classes:", np.unique(y_classificacao))
    print("Validacao cruzada: 5-fold")
    print("Learning rate do Perceptron: 0.01")
    print("Epocas do Perceptron: 100")

    for nome, parametros in modelos_classificacao:
        for padronizar in [False, True]:
            normalizacao = "Com Z-score" if padronizar else "Sem Z-score"

            print(f"\nExecutando {nome} - {normalizacao}...")

            resultado = k_fold_classificacao(
                X_classificacao,
                y_classificacao,
                k_folds=5,
                padronizar=padronizar,
                random_state=42,
                **parametros
            )

            resultados.append({
                "nome": nome,
                "padronizar": padronizar,
                "resultado": resultado
            })

            media = resultado["media"]
            desvio = resultado["desvio_padrao"]

            print("Experimento concluido.")
            print("Acuracia:", formatar_resultado(media["acuracia"], desvio["acuracia"]))
            print("Precisao:", formatar_resultado(media["precisao"], desvio["precisao"]))
            print("Recall:", formatar_resultado(media["recall"], desvio["recall"]))
            print("Especificidade:", formatar_resultado(media["especificidade"], desvio["especificidade"]))
            print("F1-Score:", formatar_resultado(media["f1_score"], desvio["f1_score"]))
            print("Tempo de treinamento (s):", formatar_resultado(media["tempo_treino"], desvio["tempo_treino"]))
            print("Tempo de teste (s):", formatar_resultado(media["tempo_teste"], desvio["tempo_teste"]))

    # Tabela comparativa
    print("\n" + "=" * 170)
    print("COMPARACAO DOS CLASSIFICADORES - MEDIA +/- DESVIO PADRAO")
    print("=" * 170)
    print(
        f"{'Modelo':<19} | {'Normalizacao':<13} | {'Acuracia':<21} | "
        f"{'Precisao':<21} | {'Recall':<21} | {'Especificidade':<21} | {'F1-Score':<21}"
    )
    print("-" * 170)

    for experimento in resultados:
        media = experimento["resultado"]["media"]
        desvio = experimento["resultado"]["desvio_padrao"]
        normalizacao = "Com Z-score" if experimento["padronizar"] else "Sem Z-score"

        print(
            f"{experimento['nome']:<19} | {normalizacao:<13} | "
            f"{formatar_resultado(media['acuracia'], desvio['acuracia']):<21} | "
            f"{formatar_resultado(media['precisao'], desvio['precisao']):<21} | "
            f"{formatar_resultado(media['recall'], desvio['recall']):<21} | "
            f"{formatar_resultado(media['especificidade'], desvio['especificidade']):<21} | "
            f"{formatar_resultado(media['f1_score'], desvio['f1_score']):<21}"
        )

    # Diferencas entre os resultados
    print("\n" + "=" * 100)
    print("IMPACTO DO Z-SCORE NA CLASSIFICACAO")
    print("Diferenca positiva significa aumento da metrica com Z-score.")
    print("=" * 100)

    metricas = ["acuracia", "precisao", "recall", "especificidade", "f1_score"]

    for nome, parametros in modelos_classificacao:
        sem = next(
            item for item in resultados
            if item["nome"] == nome and not item["padronizar"]
        )
        com = next(
            item for item in resultados
            if item["nome"] == nome and item["padronizar"]
        )

        media_sem = sem["resultado"]["media"]
        media_com = com["resultado"]["media"]

        print(f"\n{nome}:")
        for metrica in metricas:
            diferenca = media_com[metrica] - media_sem[metrica]
            print(f"{metrica}: {diferenca * 100:+.2f} pontos percentuais")

    # Matrizes de confusao do Perceptron
    print("\n" + "=" * 75)
    print("MATRIZES DE CONFUSAO - PERCEPTRON")
    print("=" * 75)

    for experimento in resultados:
        if experimento["nome"] == "Perceptron":
            resultado = experimento["resultado"]
            normalizacao = "Com Z-score" if experimento["padronizar"] else "Sem Z-score"

            print("\n" + normalizacao)
            print("Ordem das classes:", resultado["classes"])
            print(resultado["matriz_confusao"])

    return resultados


def executar_regressao():
    resultados = []

    print("\n" + "=" * 105)
    print("ANALISE DO IMPACTO DO Z-SCORE - REGRESSAO")
    print("=" * 105)
    print("Quantidade de amostras:", X_regressao.shape[0])
    print("Quantidade de atributos:", X_regressao.shape[1])
    print("Validacao cruzada: 5-fold")
    print("Observacao: o MLP padroniza internamente a variavel alvo nos dois cenarios.")

    for nome, parametros in modelos_regressao:
        for padronizar in [False, True]:
            normalizacao = "Com Z-score" if padronizar else "Sem Z-score"

            print(f"\nExecutando {nome} - {normalizacao}...")

            resultado = k_fold_regressao(
                X_regressao,
                y_regressao,
                k_folds=5,
                padronizar=padronizar,
                random_state=42,
                **parametros
            )

            resultados.append({
                "nome": nome,
                "padronizar": padronizar,
                "resultado": resultado
            })

            media = resultado["media"]
            desvio = resultado["desvio_padrao"]

            print("Experimento concluido.")
            print("MSE:", formatar_resultado(media["mse"], desvio["mse"]))
            print("RMSE:", formatar_resultado(media["rmse"], desvio["rmse"]))
            print("MAE:", formatar_resultado(media["mae"], desvio["mae"]))
            print("R2:", formatar_resultado(media["r2_score"], desvio["r2_score"]))
            print("R2 ajustado:", formatar_resultado(media["r2_score_ajus"], desvio["r2_score_ajus"]))
            print("Tempo de treinamento (s):", formatar_resultado(media["tempo_treino"], desvio["tempo_treino"]))
            print("Tempo de teste (s):", formatar_resultado(media["tempo_teste"], desvio["tempo_teste"]))

    # Tabela comparativa
    print("\n" + "=" * 175)
    print("COMPARACAO DOS REGRESSORES - MEDIA +/- DESVIO PADRAO")
    print("=" * 175)
    print(
        f"{'Modelo':<20} | {'Normalizacao':<13} | {'MSE':<21} | {'RMSE':<21} | "
        f"{'MAE':<21} | {'R2':<21} | {'R2 ajustado':<21}"
    )
    print("-" * 175)

    for experimento in resultados:
        media = experimento["resultado"]["media"]
        desvio = experimento["resultado"]["desvio_padrao"]
        normalizacao = "Com Z-score" if experimento["padronizar"] else "Sem Z-score"

        print(
            f"{experimento['nome']:<20} | {normalizacao:<13} | "
            f"{formatar_resultado(media['mse'], desvio['mse']):<21} | "
            f"{formatar_resultado(media['rmse'], desvio['rmse']):<21} | "
            f"{formatar_resultado(media['mae'], desvio['mae']):<21} | "
            f"{formatar_resultado(media['r2_score'], desvio['r2_score']):<21} | "
            f"{formatar_resultado(media['r2_score_ajus'], desvio['r2_score_ajus']):<21}"
        )

    # Diferencas entre os resultados
    print("\n" + "=" * 100)
    print("IMPACTO DO Z-SCORE NA REGRESSAO")
    print("Para RMSE/MAE, reducoes sao positivas. Para R2, aumentos sao positivos.")
    print("=" * 100)

    for nome, parametros in modelos_regressao:
        sem = next(
            item for item in resultados
            if item["nome"] == nome and not item["padronizar"]
        )
        com = next(
            item for item in resultados
            if item["nome"] == nome and item["padronizar"]
        )

        media_sem = sem["resultado"]["media"]
        media_com = com["resultado"]["media"]

        reducao_rmse = media_sem["rmse"] - media_com["rmse"]
        reducao_mae = media_sem["mae"] - media_com["mae"]
        aumento_r2 = media_com["r2_score"] - media_sem["r2_score"]

        if media_sem["rmse"] != 0:
            percentual_rmse = reducao_rmse / media_sem["rmse"] * 100
        else:
            percentual_rmse = 0.0

        print(f"\n{nome}:")
        print(f"Reducao do RMSE: {reducao_rmse:+.4f} ({percentual_rmse:+.2f}%)")
        print(f"Reducao do MAE: {reducao_mae:+.4f}")
        print(f"Variacao do R2: {aumento_r2:+.4f}")

        # O K-Fold AV3 tambem devolve metricas de treinamento.
        # Para KNN, a previsao do treino inclui as proprias amostras,
        # entao o gap treino-teste nao deve ser interpretado como uma
        # comparacao equivalente a modelos parametrizados.
        if nome == "MLP (8, ReLU)":
            mt = com["resultado"]["media_treino"]
            dt = com["resultado"]["desvio_padrao_treino"]
            me = com["resultado"]["media"]
            de = com["resultado"]["desvio_padrao"]

            print("MLP com Z-score - treino versus teste:")
            print("RMSE treino:", formatar_resultado(mt["rmse"], dt["rmse"]))
            print("RMSE teste:", formatar_resultado(me["rmse"], de["rmse"]))
            print("R2 treino:", formatar_resultado(mt["r2_score"], dt["r2_score"]))
            print("R2 teste:", formatar_resultado(me["r2_score"], de["r2_score"]))

    return resultados


# ==========================================================
# EXECUCAO PRINCIPAL
# ==========================================================

if __name__ == "__main__":
    print("ANALISE COMPARATIVA DA NORMALIZACAO Z-SCORE - AV3")
    print("Os mesmos folds (random_state=42) e parametros sao usados nos dois cenarios.")

    resultados_classificacao = executar_classificacao()
    resultados_regressao = executar_regressao()

    print("\n" + "=" * 105)
    print("FIM DA ANALISE DO Z-SCORE")
    print("=" * 105)
