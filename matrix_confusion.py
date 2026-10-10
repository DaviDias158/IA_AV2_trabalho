import numpy as np

# ALTERAÇÃO: o dataset de classificação possui 4 classes (0, 1, 2 e 3),
# então a matriz de confusão precisa funcionar para mais de duas classes.
def confusion_matrix(y_test, y_pre):
  classes = np.unique(np.concatenate((y_test, y_pre)))
  matrix = np.zeros((len(classes), len(classes)), dtype=int)

  # Linhas = classe real; colunas = classe prevista.
  for real, pred in zip(y_test, y_pre):
    i = np.where(classes == real)[0][0]
    j = np.where(classes == pred)[0][0]
    matrix[i, j] += 1

  return matrix, classes

# ALTERAÇÃO: em vez de receber VP, VN, FP e FN prontos (caso binário),
# as funções recebem os rótulos reais e previstos e calculam as métricas.
def accuracy(y_test, y_pre):
  return np.mean(np.asarray(y_test) == np.asarray(y_pre))

# ALTERAÇÃO: em multiclasse, precision, recall e F1 são calculados por classe.
# Por padrão usamos macro-média: cada classe tem o mesmo peso.
def precision(y_test, y_pre):
  matrix, classes = confusion_matrix(y_test, y_pre)
  precisions = []

  for i, c in enumerate(classes):
    VP = matrix[i, i]
    FP = np.sum(matrix[:, i]) - VP
    precisions.append(VP / (VP + FP) if (VP + FP) > 0 else 0)

  return np.mean(precisions)


def recall(y_test, y_pre):
  matrix, classes = confusion_matrix(y_test, y_pre)
  recalls = []

  for i, c in enumerate(classes):
    VP = matrix[i, i]
    FN = np.sum(matrix[i, :]) - VP
    recalls.append(VP / (VP + FN) if (VP + FN) > 0 else 0)

  return np.mean(recalls)


def f1_score(y_test, y_pre):
  matrix, classes = confusion_matrix(y_test, y_pre)
  f1_scores = []

  for i, c in enumerate(classes):
    VP = matrix[i, i]
    FP = np.sum(matrix[:, i]) - VP
    FN = np.sum(matrix[i, :]) - VP

    p = VP / (VP + FP) if (VP + FP) > 0 else 0
    r = VP / (VP + FN) if (VP + FN) > 0 else 0
    f1_scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0)

  return np.mean(f1_scores)
