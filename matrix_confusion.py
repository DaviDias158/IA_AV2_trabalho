import numpy as np

def confusion_matrix(y_test, y_pre):
  classes = np.unique(np.concatenate((y_test, y_pre)))
  matrix = np.zeros((len(classes), len(classes)), dtype=int)

  for real, pred in zip(y_test, y_pre):
    i = np.where(classes == real)[0][0]
    j = np.where(classes == pred)[0][0]
    matrix[i, j] += 1

  return matrix, classes

def accuracy(y_test, y_pre):
  return np.mean(np.asarray(y_test) == np.asarray(y_pre))

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
