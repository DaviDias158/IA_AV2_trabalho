
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


# Arquiteturas selecionadas
configuracoes = [
    ((16,), "tanh"),
    ((8,), "relu")
]

# Taxas de aprendizagem
taxas = [0.001, 0.01, 0.03, 0.1]

epocas = 300

resultados = []


print("EXPERIMENTOS - TAXA DE APRENDIZAGEM MLP")
print("Amostras:", len(X))
print("Atributos:", X.shape[1])
print("Validação cruzada: 5 folds")


# Executa as combinações
for topologia, ativacao in configuracoes:
    for taxa in taxas:

        print(
            f"\nExecutando: topologia={topologia}, "
            f"ativacao={ativacao}, "
            f"learning_rate={taxa}"
        )

        resultado = k_fold_regressao(
            X,
            y,
            modelo="mlp",
            k_folds=5,
            hidden_layers=topologia,
            activation=ativacao,
            learning_rate=taxa,
            epochs=epocas,
            padronizar=True,
            random_state=42
        )

        media = resultado["media"]

        resultados.append({
            "topologia": topologia,
            "ativacao": ativacao,
            "taxa": taxa,
            "resultado": resultado
        })

        print(f"RMSE: {media['rmse']:.4f}")
        print(f"MAE: {media['mae']:.4f}")
        print(f"R2: {media['r2_score']:.4f}")
        print(f"Treino: {media['tempo_treino']:.4f} s")


# Tabela comparativa
print("\n" + "=" * 110)
print("COMPARAÇÃO DAS TAXAS DE APRENDIZAGEM")
print("=" * 110)

print(
    f"{'Topologia':<12} | {'Ativação':<9} | "
    f"{'Taxa':<8} | {'MSE':<10} | "
    f"{'RMSE':<10} | {'MAE':<10} | "
    f"{'R2':<9} | {'Treino (s)':<10}"
)

print("-" * 110)

for experimento in resultados:
    media = experimento["resultado"]["media"]

    print(
        f"{str(experimento['topologia']):<12} | "
        f"{experimento['ativacao']:<9} | "
        f"{experimento['taxa']:<8} | "
        f"{media['mse']:<10.4f} | "
        f"{media['rmse']:<10.4f} | "
        f"{media['mae']:<10.4f} | "
        f"{media['r2_score']:<9.4f} | "
        f"{media['tempo_treino']:<10.4f}"
    )


# Escolhe a configuração com menor RMSE
melhor = min(
    resultados,
    key=lambda exp: exp["resultado"]["media"]["rmse"]
)

resultado_melhor = melhor["resultado"]

print("\n" + "=" * 60)
print("MELHOR CONFIGURAÇÃO")
print("=" * 60)

print("Topologia:", melhor["topologia"])
print("Ativação:", melhor["ativacao"])
print("Taxa:", melhor["taxa"])
print("Épocas:", epocas)

print("\nMÉTRICAS - MÉDIA ± DESVIO PADRÃO")

for metrica, valor in resultado_melhor["media"].items():
    desvio = resultado_melhor["desvio_padrao"][metrica]

    print(f"{metrica}: {valor:.4f} ± {desvio:.4f}")
