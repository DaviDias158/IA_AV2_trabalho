
import numpy as np
from itertools import product

from k_fold import k_fold_regressao


# Leitura da base de regressão
dados = np.loadtxt(
    "cpu_small_227.csv",
    delimiter=",",
    skiprows=1
)

X = dados[:, 1:-1]
y = dados[:, -1]


# Configurações dos experimentos
topologias = [
    (8,),
    (16,),
    (16, 8)
]

ativacoes = ["tanh", "relu"]
numeros_epocas = [100, 300]

# Taxa fixa nesta primeira etapa
taxa_aprendizagem = 0.01

resultados = []


print("EXPERIMENTOS - MLP REGRESSÃO")
print("Amostras:", len(X))
print("Atributos:", X.shape[1])
print("Validação cruzada: 5 folds")


# Executa todas as combinações
for topologia, ativacao, epocas in product(
    topologias,
    ativacoes,
    numeros_epocas
):

    print(
        f"\nExecutando: topologia={topologia}, "
        f"ativacao={ativacao}, epochs={epocas}"
    )

    resultado = k_fold_regressao(
        X,
        y,
        modelo="mlp",
        k_folds=5,
        hidden_layers=topologia,
        activation=ativacao,
        learning_rate=taxa_aprendizagem,
        epochs=epocas,
        padronizar=True,
        random_state=42
    )

    media = resultado["media"]

    resultados.append({
        "topologia": topologia,
        "ativacao": ativacao,
        "epocas": epocas,
        "resultado": resultado
    })

    print(f"RMSE: {media['rmse']:.4f}")
    print(f"MAE: {media['mae']:.4f}")
    print(f"R2: {media['r2_score']:.4f}")
    print(f"Tempo treino: {media['tempo_treino']:.4f} s")


# Exibe a tabela comparativa
print("\n" + "=" * 105)
print("COMPARAÇÃO DAS ARQUITETURAS")
print("=" * 105)

print(
    f"{'Topologia':<13} | {'Ativação':<9} | "
    f"{'Épocas':<7} | {'MSE':<10} | "
    f"{'RMSE':<10} | {'MAE':<10} | "
    f"{'R2':<9} | {'Treino (s)':<10}"
)

print("-" * 105)

for experimento in resultados:
    media = experimento["resultado"]["media"]

    print(
        f"{str(experimento['topologia']):<13} | "
        f"{experimento['ativacao']:<9} | "
        f"{experimento['epocas']:<7} | "
        f"{media['mse']:<10.4f} | "
        f"{media['rmse']:<10.4f} | "
        f"{media['mae']:<10.4f} | "
        f"{media['r2_score']:<9.4f} | "
        f"{media['tempo_treino']:<10.4f}"
    )


# Seleciona a configuração com menor RMSE
melhor = min(
    resultados,
    key=lambda exp: exp["resultado"]["media"]["rmse"]
)

resultado_melhor = melhor["resultado"]

print("\n" + "=" * 60)
print("MELHOR CONFIGURAÇÃO PELO RMSE")
print("=" * 60)

print("Topologia:", melhor["topologia"])
print("Ativação:", melhor["ativacao"])
print("Épocas:", melhor["epocas"])
print("Learning rate:", taxa_aprendizagem)

print("\nMÉTRICAS - MÉDIA ± DESVIO PADRÃO")

for metrica, valor in resultado_melhor["media"].items():
    desvio = resultado_melhor["desvio_padrao"][metrica]

    print(f"{metrica}: {valor:.4f} ± {desvio:.4f}")
