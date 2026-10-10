import numpy as np

from k_fold import k_fold_classificacao, k_fold_regressao

#Leitura
dados_regressao = np.loadtxt("cpu_small_227.csv", delimiter=",", skiprows=1)
X_regressao = dados_regressao[:, 1:-1]
y_regressao = dados_regressao[:, -1]

dados_classificacao = np.loadtxt("football_player_position_46763.csv", delimiter=",", skiprows=38)
X_classificacao = dados_classificacao[:, 1:-1]
y_classificacao = dados_classificacao[:, -1].astype(int) #.astype(int) serva pra deixar tudo int na classificação, testar com e sem ele

#print("Dados de regressão:", X_regressao.shape, y_regressao.shape)
#print("Dados de classificação:", X_classificacao.shape, y_classificacao.shape)
#print("Classes encontradas:", np.unique(y_classificacao))

#item a)
print("\na)")
resultado_knn_euclidiana = k_fold_classificacao(
    X_classificacao,
    y_classificacao,
    modelo="knn",
    k_folds=5,
    distancia="euclidean",
)
print("Teste com distancia Euclidiana concluido.")

resultado_knn_manhattan = k_fold_classificacao(
    X_classificacao,
    y_classificacao,
    modelo="knn",
    k_folds=5,
    distancia="manhattan",
)
print("Teste com distancia Manhattan concluido.")

#item b)
print("\nb)")
resultado_naive_bayes = k_fold_classificacao(
    X_classificacao,
    y_classificacao,
    modelo="naive_bayes",
    k_folds=5,
)
print("Teste do Naive Bayes univariado concluído.")

#item d)
print("\nd)")
def formatar_resultado(media, desvio):
    return f"{media:.3f} ± {desvio:.3f}"


resultados_classificadores = [
    ("K-vizinhos (dist. Euclidiana)", resultado_knn_euclidiana),
    ("K-vizinhos (dist. Manhattan)", resultado_knn_manhattan),
    ("Bayesiano (Univariado)", resultado_naive_bayes),
]
print(
    f"{'Classificador':<31} | {'Acurácia':<17} | {'Precisão':<17} | "
    f"{'F1-Score':<17} | {'Tempo Treino (s)':<19} | {'Tempo Teste (s)':<18}"
)
print("-" * 133)

for nome, resultado in resultados_classificadores:
    media = resultado["media"]
    desvio = resultado["desvio_padrao"]

    print(
        f"{nome:<31} | "
        f"{formatar_resultado(media['acuracia'], desvio['acuracia']):<17} | "
        f"{formatar_resultado(media['precisao'], desvio['precisao']):<17} | "
        f"{formatar_resultado(media['f1_score'], desvio['f1_score']):<17} | "
        f"{formatar_resultado(media['tempo_treino'], desvio['tempo_treino']):<19} | "
        f"{formatar_resultado(media['tempo_teste'], desvio['tempo_teste']):<18}"
    )

#item e)
# Para regressao, usamos R2, R2 ajustado e MAE; acuracia, precisao,
# recall e F1-score sao metricas de classificacao.
print("\ne)")
resultado_knn_reg_euclidiana = k_fold_regressao(
    X_regressao,
    y_regressao,
    modelo="knn",
    k_folds=5,
    distancia="euclidean",
)
print("Teste de regressao com distancia Euclidiana concluido.")

resultado_knn_reg_manhattan = k_fold_regressao(
    X_regressao,
    y_regressao,
    modelo="knn",
    k_folds=5,
    distancia="manhattan",
)
print("Teste de regressao com distancia Manhattan concluido.")

resultado_regressao_linear = k_fold_regressao(
    X_regressao,
    y_regressao,
    modelo="linear_regression",
    k_folds=5,
)
print("Teste de regressao linear multipla concluido.")

resultados_regressores = [
    ("KNN (dist. Euclidiana)", resultado_knn_reg_euclidiana),
    ("KNN (dist. Manhattan)", resultado_knn_reg_manhattan),
    ("Regressao linear multipla", resultado_regressao_linear),
]

print("\nComparacao dos regressores (5-fold)")
print(
    f"{'Regressor':<28} | {'R2':<17} | {'R2 ajustado':<17} | "
    f"{'MAE':<17} | {'Tempo Treino (s)':<19} | {'Tempo Teste (s)':<18}"
)
print("-" * 128)

# ALTERACAO: imprime uma linha numerica para cada um dos tres regressores.
for nome, resultado in resultados_regressores:
    media = resultado["media"]
    desvio = resultado["desvio_padrao"]

    r2 = formatar_resultado(media["r2_score"], desvio["r2_score"])
    r2_ajustado = formatar_resultado(
        media["r2_score_ajus"], desvio["r2_score_ajus"]
    )
    mae = formatar_resultado(media["mae"], desvio["mae"])
    tempo_treino = formatar_resultado(
        media["tempo_treino"], desvio["tempo_treino"]
    )
    tempo_teste = formatar_resultado(
        media["tempo_teste"], desvio["tempo_teste"]
    )

    print(
        f"{nome:<28} | {r2:<17} | {r2_ajustado:<17} | "
        f"{mae:<17} | {tempo_treino:<19} | {tempo_teste:<18}"
    )
