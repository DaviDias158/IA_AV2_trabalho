import numpy as np

from k_fold import k_fold_classificacao


# ==========================================================
# LEITURA DO DATASET DE CLASSIFICACAO
# ==========================================================

dados_classificacao = np.loadtxt(
    "football_player_position_46763.csv",
    delimiter=",",
    skiprows=38
)

X_classificacao = dados_classificacao[:, 1:-1]
y_classificacao = dados_classificacao[:, -1].astype(int)


# ==========================================================
# CONFIGURACOES DOS EXPERIMENTOS
# ==========================================================

# Serão testadas todas as combinacoes entre as taxas e o numero de epocas.
taxas_aprendizagem = [0.001, 0.01, 0.1]
numeros_epocas = [10, 30, 100]

# Armazena os resultados de cada combinacao.
experimentos = []


print("=" * 75)
print("CLASSIFICACAO - PERCEPTRON SIMPLES MULTICLASSE")
print("=" * 75)
print("Quantidade de amostras:", X_classificacao.shape[0])
print("Quantidade de atributos:", X_classificacao.shape[1])
print("Classes encontradas:", np.unique(y_classificacao))
print("Validacao cruzada: 5-fold")
print("Padronizacao Z-score: ativada")


# ==========================================================
# EXECUCAO DOS EXPERIMENTOS
# ==========================================================

for taxa in taxas_aprendizagem:
    for epocas in numeros_epocas:

        print(
            "\nExecutando experimento: "
            f"learning_rate={taxa}, epochs={epocas}"
        )

        resultado = k_fold_classificacao(
            X_classificacao,
            y_classificacao,
            modelo="perceptron",
            k_folds=5,
            learning_rate=taxa,
            epochs=epocas,
            padronizar=True,
            random_state=42
        )

        experimentos.append({
            "taxa": taxa,
            "epocas": epocas,
            "resultado": resultado
        })

        print("Experimento concluido.")


# ==========================================================
# COMPARACAO DOS RESULTADOS
# ==========================================================

def formatar_resultado(media, desvio):
    return f"{media:.4f} +/- {desvio:.4f}"


print("\n" + "=" * 150)
print("COMPARACAO ENTRE AS CONFIGURACOES DO PERCEPTRON")
print("=" * 150)

print(
    f"{'Learning rate':<15} | {'Epocas':<7} | "
    f"{'Acuracia':<20} | {'Recall':<20} | "
    f"{'Especificidade':<20} | {'Precisao':<20} | "
    f"{'F1-Score':<20} | {'Tempo treino (s)':<20} | "
    f"{'Tempo teste (s)':<20}"
)
print("-" * 150)

for experimento in experimentos:
    media = experimento["resultado"]["media"]
    desvio = experimento["resultado"]["desvio_padrao"]

    print(
        f"{experimento['taxa']:<15} | "
        f"{experimento['epocas']:<7} | "
        f"{formatar_resultado(media['acuracia'], desvio['acuracia']):<20} | "
        f"{formatar_resultado(media['recall'], desvio['recall']):<20} | "
        f"{formatar_resultado(media['especificidade'], desvio['especificidade']):<20} | "
        f"{formatar_resultado(media['precisao'], desvio['precisao']):<20} | "
        f"{formatar_resultado(media['f1_score'], desvio['f1_score']):<20} | "
        f"{formatar_resultado(media['tempo_treino'], desvio['tempo_treino']):<20} | "
        f"{formatar_resultado(media['tempo_teste'], desvio['tempo_teste']):<20}"
    )


# ==========================================================
# SELECAO DA MELHOR CONFIGURACAO
# ==========================================================

# O F1-Score macro e o criterio principal.
# Em caso de empate, a acuracia e usada como criterio de desempate.
melhor = max(
    experimentos,
    key=lambda experimento: (
        experimento["resultado"]["media"]["f1_score"],
        experimento["resultado"]["media"]["acuracia"]
    )
)

resultado_melhor = melhor["resultado"]
media_melhor = resultado_melhor["media"]
desvio_melhor = resultado_melhor["desvio_padrao"]

print("\n" + "=" * 75)
print("MELHOR CONFIGURACAO - CRITERIO: F1-SCORE")
print("=" * 75)
print("Learning rate:", melhor["taxa"])
print("Numero de epocas:", melhor["epocas"])

print("\nMetricas da melhor configuracao (media +/- desvio padrao):")
print("Acuracia:", formatar_resultado(media_melhor["acuracia"], desvio_melhor["acuracia"]))
print("Sensibilidade (Recall):", formatar_resultado(media_melhor["recall"], desvio_melhor["recall"]))
print("Especificidade:", formatar_resultado(media_melhor["especificidade"], desvio_melhor["especificidade"]))
print("Precisao (Precision):", formatar_resultado(media_melhor["precisao"], desvio_melhor["precisao"]))
print("F1-Score:", formatar_resultado(media_melhor["f1_score"], desvio_melhor["f1_score"]))
print("Tempo de treinamento (s):", formatar_resultado(media_melhor["tempo_treino"], desvio_melhor["tempo_treino"]))
print("Tempo de teste (s):", formatar_resultado(media_melhor["tempo_teste"], desvio_melhor["tempo_teste"]))


# ==========================================================
# MATRIZ DE CONFUSAO DA MELHOR CONFIGURACAO
# ==========================================================

print("\n" + "=" * 75)
print("MATRIZ DE CONFUSAO - MELHOR CONFIGURACAO")
print("=" * 75)
print("Ordem das classes (linhas = reais; colunas = previstas):")
print(resultado_melhor["classes"])
print(resultado_melhor["matriz_confusao"])
print("Total de amostras avaliadas:", np.sum(resultado_melhor["matriz_confusao"]))
