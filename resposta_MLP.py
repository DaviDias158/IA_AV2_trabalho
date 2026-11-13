
import numpy as np

from k_fold_AV3 import k_fold_regressao


# ==========================================================
# LEITURA DO DATASET DE REGRESSAO
# ==========================================================

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

# Etapa 1: compara topologias, funcoes de ativacao e epocas.
# A taxa de aprendizagem fica fixa para isolar o efeito desses fatores.
topologias = [
    (8,),
    (16,),
    (16, 8)
]

funcoes_ativacao = ["tanh", "relu"]
numeros_epocas = [100, 300]
taxa_base = 0.01

configuracoes = []

for topologia in topologias:
    for ativacao in funcoes_ativacao:
        for epocas in numeros_epocas:
            configuracoes.append({
                "topologia": topologia,
                "ativacao": ativacao,
                "taxa": taxa_base,
                "epocas": epocas,
                "etapa": "Arquitetura"
            })


# Etapa 2: investiga o efeito da taxa de aprendizagem.
# O teste e feito em duas arquiteturas de referencia, com 300 epocas.
# A taxa 0.01 ja foi executada na etapa anterior, evitando repeticao.
configuracoes_taxa = [
    ((16,), "tanh"),
    ((8,), "relu")
]

taxas_adicionais = [0.001, 0.03, 0.1]

for topologia, ativacao in configuracoes_taxa:
    for taxa in taxas_adicionais:
        configuracoes.append({
            "topologia": topologia,
            "ativacao": ativacao,
            "taxa": taxa,
            "epocas": 300,
            "etapa": "Taxa de aprendizagem"
        })


# Armazena os resultados de todas as configuracoes.
experimentos = []


def formatar_resultado(media, desvio):
    return f"{media:.4f} +/- {desvio:.4f}"


# ==========================================================
# INFORMACOES INICIAIS
# ==========================================================

print("=" * 90)
print("REGRESSAO - MULTI LAYER PERCEPTRON (MLP)")
print("=" * 90)
print("Quantidade de amostras:", X_regressao.shape[0])
print("Quantidade de atributos:", X_regressao.shape[1])
print("Validacao cruzada: 5-fold")
print("Padronizacao Z-score das entradas: ativada")
print("Padronizacao da variavel alvo: realizada internamente pelo MLP")
print("Metricas calculadas: treinamento e teste")
print("Quantidade de experimentos:", len(configuracoes))


# ==========================================================
# EXECUCAO DOS EXPERIMENTOS
# ==========================================================

for indice, configuracao in enumerate(configuracoes, start=1):

    topologia = configuracao["topologia"]
    ativacao = configuracao["ativacao"]
    taxa = configuracao["taxa"]
    epocas = configuracao["epocas"]

    print("\n" + "-" * 90)
    print(f"Experimento {indice}/{len(configuracoes)} - {configuracao['etapa']}")
    print(
        f"Topologia={topologia}, ativacao={ativacao}, "
        f"learning_rate={taxa}, epochs={epocas}"
    )

    resultado = k_fold_regressao(
        X_regressao,
        y_regressao,
        modelo="mlp",
        k_folds=5,
        hidden_layers=topologia,
        activation=ativacao,
        learning_rate=taxa,
        epochs=epocas,
        padronizar=True,
        random_state=42
    )

    experimentos.append({
        "topologia": topologia,
        "ativacao": ativacao,
        "taxa": taxa,
        "epocas": epocas,
        "etapa": configuracao["etapa"],
        "resultado": resultado
    })

    media = resultado["media"]
    desvio = resultado["desvio_padrao"]
    media_treino = resultado["media_treino"]
    desvio_treino = resultado["desvio_padrao_treino"]

    print("Experimento concluido.")

    print("\nRESULTADOS DE TREINAMENTO:")
    print("MSE treino:", formatar_resultado(media_treino["mse"], desvio_treino["mse"]))
    print("RMSE treino:", formatar_resultado(media_treino["rmse"], desvio_treino["rmse"]))
    print("MAE treino:", formatar_resultado(media_treino["mae"], desvio_treino["mae"]))
    print("R2 treino:", formatar_resultado(media_treino["r2_score"], desvio_treino["r2_score"]))
    print("R2 ajustado treino:", formatar_resultado(media_treino["r2_score_ajus"], desvio_treino["r2_score_ajus"]))

    print("\nRESULTADOS DE TESTE:")
    print("MSE teste:", formatar_resultado(media["mse"], desvio["mse"]))
    print("RMSE teste:", formatar_resultado(media["rmse"], desvio["rmse"]))
    print("MAE teste:", formatar_resultado(media["mae"], desvio["mae"]))
    print("R2 teste:", formatar_resultado(media["r2_score"], desvio["r2_score"]))
    print("R2 ajustado teste:", formatar_resultado(media["r2_score_ajus"], desvio["r2_score_ajus"]))

    print(
        "\nTempo de treinamento (s):",
        formatar_resultado(media["tempo_treino"], desvio["tempo_treino"])
    )

    print(
        "Tempo de teste (s):",
        formatar_resultado(media["tempo_teste"], desvio["tempo_teste"])
    )


# ==========================================================
# TABELA COMPARATIVA - ERROS DE TESTE
# ==========================================================

print("\n" + "=" * 145)
print("COMPARACAO DOS EXPERIMENTOS - MSE, RMSE E MAE (TESTE)")
print("=" * 145)

print(
    f"{'Topologia':<13} | {'Ativacao':<8} | {'Taxa':<7} | "
    f"{'Epocas':<6} | {'MSE (media +/- DP)':<24} | "
    f"{'RMSE (media +/- DP)':<24} | {'MAE (media +/- DP)':<24}"
)
print("-" * 145)

for experimento in experimentos:
    media = experimento["resultado"]["media"]
    desvio = experimento["resultado"]["desvio_padrao"]

    print(
        f"{str(experimento['topologia']):<13} | "
        f"{experimento['ativacao']:<8} | "
        f"{experimento['taxa']:<7} | "
        f"{experimento['epocas']:<6} | "
        f"{formatar_resultado(media['mse'], desvio['mse']):<24} | "
        f"{formatar_resultado(media['rmse'], desvio['rmse']):<24} | "
        f"{formatar_resultado(media['mae'], desvio['mae']):<24}"
    )


# ==========================================================
# TABELA COMPARATIVA - R2 E TEMPOS
# ==========================================================

print("\n" + "=" * 145)
print("COMPARACAO DOS EXPERIMENTOS - R2 E CUSTO COMPUTACIONAL")
print("=" * 145)

print(
    f"{'Topologia':<13} | {'Ativacao':<8} | {'Taxa':<7} | "
    f"{'Epocas':<6} | {'R2 (media +/- DP)':<23} | "
    f"{'R2 ajust. (media +/- DP)':<25} | "
    f"{'Treino (s)':<23} | {'Teste (s)':<23}"
)
print("-" * 145)

for experimento in experimentos:
    media = experimento["resultado"]["media"]
    desvio = experimento["resultado"]["desvio_padrao"]

    print(
        f"{str(experimento['topologia']):<13} | "
        f"{experimento['ativacao']:<8} | "
        f"{experimento['taxa']:<7} | "
        f"{experimento['epocas']:<6} | "
        f"{formatar_resultado(media['r2_score'], desvio['r2_score']):<23} | "
        f"{formatar_resultado(media['r2_score_ajus'], desvio['r2_score_ajus']):<25} | "
        f"{formatar_resultado(media['tempo_treino'], desvio['tempo_treino']):<23} | "
        f"{formatar_resultado(media['tempo_teste'], desvio['tempo_teste']):<23}"
    )


# ==========================================================
# TABELA - COMPARACAO TREINAMENTO X TESTE
# ==========================================================

print("\n" + "=" * 175)
print("COMPARACAO TREINAMENTO X TESTE - ANALISE DE GENERALIZACAO")
print("=" * 175)

print(
    f"{'Topologia':<11} | {'Ativacao':<7} | {'Taxa':<6} | {'Epocas':<6} | "
    f"{'RMSE treino':<13} | {'RMSE teste':<13} | {'MAE treino':<12} | "
    f"{'MAE teste':<12} | {'R2 treino':<12} | {'R2 teste':<12} | {'Gap R2':<10}"
)
print("-" * 175)

for experimento in experimentos:
    resultado = experimento["resultado"]

    mt = resultado["media_treino"]
    dt = resultado["desvio_padrao_treino"]

    me = resultado["media"]
    de = resultado["desvio_padrao"]

    gap_r2 = mt["r2_score"] - me["r2_score"]

    print(
        f"{str(experimento['topologia']):<11} | "
        f"{experimento['ativacao']:<7} | "
        f"{experimento['taxa']:<6} | "
        f"{experimento['epocas']:<6} | "
        f"{formatar_resultado(mt['rmse'], dt['rmse']):<13} | "
        f"{formatar_resultado(me['rmse'], de['rmse']):<13} | "
        f"{formatar_resultado(mt['mae'], dt['mae']):<12} | "
        f"{formatar_resultado(me['mae'], de['mae']):<12} | "
        f"{formatar_resultado(mt['r2_score'], dt['r2_score']):<12} | "
        f"{formatar_resultado(me['r2_score'], de['r2_score']):<12} | "
        f"{gap_r2:.4f}"
    )


# ==========================================================
# SELECAO DA MELHOR CONFIGURACAO
# ==========================================================

# O RMSE do conjunto de teste e o criterio principal: quanto menor, melhor.
# Em caso de empate, escolhe-se o maior R2 medio de teste.

melhor = min(
    experimentos,
    key=lambda experimento: (
        experimento["resultado"]["media"]["rmse"],
        -experimento["resultado"]["media"]["r2_score"]
    )
)

resultado_melhor = melhor["resultado"]

media_melhor = resultado_melhor["media"]
desvio_melhor = resultado_melhor["desvio_padrao"]

media_treino_melhor = resultado_melhor["media_treino"]
desvio_treino_melhor = resultado_melhor["desvio_padrao_treino"]

gap_rmse = media_melhor["rmse"] - media_treino_melhor["rmse"]
gap_r2 = media_treino_melhor["r2_score"] - media_melhor["r2_score"]


print("\n" + "=" * 90)
print("MELHOR CONFIGURACAO - CRITERIO: MENOR RMSE DE TESTE")
print("=" * 90)

print("Topologia:", melhor["topologia"])
print("Quantidade de camadas ocultas:", len(melhor["topologia"]))
print("Neuronios por camada:", melhor["topologia"])
print("Funcao de ativacao:", melhor["ativacao"])
print("Learning rate:", melhor["taxa"])
print("Numero de epocas:", melhor["epocas"])


print("\nMETRICAS DE TREINAMENTO - MEDIA +/- DESVIO PADRAO")

print("MSE:", formatar_resultado(media_treino_melhor["mse"], desvio_treino_melhor["mse"]))
print("RMSE:", formatar_resultado(media_treino_melhor["rmse"], desvio_treino_melhor["rmse"]))
print("MAE:", formatar_resultado(media_treino_melhor["mae"], desvio_treino_melhor["mae"]))
print("R2 Score:", formatar_resultado(media_treino_melhor["r2_score"], desvio_treino_melhor["r2_score"]))
print("R2 Ajustado:", formatar_resultado(media_treino_melhor["r2_score_ajus"], desvio_treino_melhor["r2_score_ajus"]))


print("\nMETRICAS DE TESTE - MEDIA +/- DESVIO PADRAO")

print("MSE:", formatar_resultado(media_melhor["mse"], desvio_melhor["mse"]))
print("RMSE:", formatar_resultado(media_melhor["rmse"], desvio_melhor["rmse"]))
print("MAE:", formatar_resultado(media_melhor["mae"], desvio_melhor["mae"]))
print("R2 Score:", formatar_resultado(media_melhor["r2_score"], desvio_melhor["r2_score"]))
print("R2 Ajustado:", formatar_resultado(media_melhor["r2_score_ajus"], desvio_melhor["r2_score_ajus"]))

print(
    "Tempo de treinamento (s):",
    formatar_resultado(media_melhor["tempo_treino"], desvio_melhor["tempo_treino"])
)

print(
    "Tempo de teste (s):",
    formatar_resultado(media_melhor["tempo_teste"], desvio_melhor["tempo_teste"])
)


print("\nDIFERENCA ENTRE TREINAMENTO E TESTE")
print(f"Diferenca de RMSE (teste - treino): {gap_rmse:.4f}")
print(f"Diferenca de R2 (treino - teste): {gap_r2:.4f}")


# ==========================================================
# OBSERVACOES PARA A ANALISE DOS RESULTADOS
# ==========================================================

print("\n" + "=" * 90)
print("OBSERVACOES SOBRE GENERALIZACAO E CUSTO COMPUTACIONAL")
print("=" * 90)

print(
    "A validacao cruzada avalia o desempenho em amostras que nao participaram "
    "do treinamento de cada fold."
)

print(
    "Agora tambem sao calculadas metricas nas amostras de treinamento de cada fold. "
    "A comparacao entre treino e teste ajuda a investigar possiveis sinais de overfitting."
)

print(
    "Uma diferenca pequena entre as metricas de treino e teste e um indicio favoravel "
    "de generalizacao, mas deve ser interpretada junto com a variacao entre os folds."
)

print(
    "Os desvios padrao das metricas mostram a variacao do desempenho entre os folds; "
    "valores menores indicam resultados mais consistentes."
)

print(
    "O tempo medio de treinamento e o tempo medio de teste permitem comparar "
    "o custo computacional das configuracoes."
)
