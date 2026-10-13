import numpy as np

class NaiveBayes: # Univariado
    def fit(self, X_train, y_train):
      n_samples, n_features = X_train.shape
      self._classes = np.unique(y_train)
      n_classes = len(self._classes)
      self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
      self._var = np.zeros((n_classes, n_features), dtype=np.float64)
      self._prior = np.zeros(n_classes, dtype=np.float64)

      for idx, c in enumerate(self._classes):
        X_c = X_train[y_train == c]
        self._mean[idx, :] = X_c.mean(axis=0)
        self._var[idx, :] = np.maximum(X_c.var(axis=0), 1e-9)
        self._prior[idx] = X_c.shape[0] / float(n_samples)

    def _predict(self, x):
      posteriors = []
      for idx, c in enumerate(self._classes):
        prior = np.log(self._prior[idx])
        posterior = np.sum(
            -0.5 * np.log(2 * np.pi * self._var[idx])
            - ((x - self._mean[idx]) ** 2) / (2 * self._var[idx])
        )
        posterior = prior + posterior
        posteriors.append(posterior)
      return self._classes[np.argmax(posteriors)]

    def predict(self, X_test):
      y_pred = [self._predict(x) for x in X_test]
      return np.array(y_pred)

    def _pdf(self, class_idx, x):
      mean = self._mean[class_idx]
      var = self._var[class_idx]
      numerator = np.exp(-((x - mean) ** 2) / (2 * var))
      denominator = np.sqrt(2 * np.pi * var)
      return numerator / denominator
