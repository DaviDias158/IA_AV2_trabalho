
import numpy as np

from k_fold import k_fold_classificacao


# Leitura da base de classificação
dados = np.loadtxt(
    "football_player_position_46763.csv",
    delimiter=",",
    skiprows=38
)

X = dados[:, 1:-1]
y = dados[:, -1].astype(int)


# Hiperparâmetros que serão avaliados
taxas_aprendizagem = [0.001, 0.01, 0.1]
numeros_epocas = [10, 30, 100]


# Armazena os resultados dos experimentos
experimentos = []


print("EXPERIMENTOS - PERCEPTRON MULTICLASSE")
print("Amostras:", len(X))
print("Atributos:", X.shape[1])
print("Classes:", np.unique(y))
print("Validação cruzada: 5 folds")


# Executa todas as combinações de hiperparâmetros
for taxa in taxas_aprendizagem:
    for epocas in numeros_epocas:

        print(
            f"\nExecutando: learning_rate={taxa}, "
            f"epochs={epocas}"
        )

        resultado = k_fold_classificacao(
            X,
            y,
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

        print("Experimento concluído.")


# Tabela comparativa dos resultados
print("\n" + "=" * 115)
print("COMPARAÇÃO DOS HIPERPARÂMETROS")
print("=" * 115)

print(
    f"{'Taxa':<8} | {'Épocas':<7} | "
    f"{'Acurácia':<10} | {'Precisão':<10} | "
    f"{'Recall':<10} | {'Especif.':<10} | "
    f"{'F1-Score':<10} | {'Treino (s)':<12}"
)

print("-" * 115)

for experimento in experimentos:

    media = experimento["resultado"]["media"]

    print(
        f"{experimento['taxa']:<8} | "
        f"{experimento['epocas']:<7} | "
        f"{media['acuracia']:<10.4f} | "
        f"{media['precisao']:<10.4f} | "
        f"{media['recall']:<10.4f} | "
        f"{media['especificidade']:<10.4f} | "
        f"{media['f1_score']:<10.4f} | "
        f"{media['tempo_treino']:<12.4f}"
    )


# Seleciona a configuração de maior F1-Score macro.
# A acurácia é usada como critério de desempate.
melhor = max(
    experimentos,
    key=lambda exp: (
        exp["resultado"]["media"]["f1_score"],
        exp["resultado"]["media"]["acuracia"]
    )
)

resultado_melhor = melhor["resultado"]

print("\n" + "=" * 60)
print("MELHOR CONFIGURAÇÃO PELO F1-SCORE")
print("=" * 60)

print("Taxa de aprendizagem:", melhor["taxa"])
print("Número de épocas:", melhor["epocas"])

print("\nMétricas - média ± desvio padrão")

for metrica, valor in resultado_melhor["media"].items():
    desvio = resultado_melhor["desvio_padrao"][metrica]

    print(f"{metrica}: {valor:.4f} ± {desvio:.4f}")


# Matriz de confusão da configuração selecionada
print("\nMATRIZ DE CONFUSÃO")

print("Classes:", resultado_melhor["classes"])
print(resultado_melhor["matriz_confusao"])

print(
    "\nTotal de amostras:",
    np.sum(resultado_melhor["matriz_confusao"])
)
