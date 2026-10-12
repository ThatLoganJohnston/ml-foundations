from __future__ import annotations

from collections import Counter

import numpy as np

from ._validation import as_feature_matrix, as_target_vector, require_fitted


class KNNClassifier:
    def __init__(self, neighbors: int = 5, metric: str = "euclidean") -> None:
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        self.neighbors = neighbors
        self.metric = metric

    def fit(self, X: object, y: object) -> KNNClassifier:
        features = as_feature_matrix(X)
        if self.neighbors > len(features):
            raise ValueError("neighbors cannot exceed the number of training samples")
        self.X_ = features
        self.y_ = as_target_vector(y, len(features))
        return self

    def predict(self, X: object) -> np.ndarray:
        require_fitted(self, "X_")
        features = as_feature_matrix(X)
        if features.shape[1] != self.X_.shape[1]:
            raise ValueError("feature count does not match the fitted data")

        predictions = []
        for row in features:
            difference = np.abs(self.X_ - row)
            distances = (
                np.sqrt(np.square(difference).sum(axis=1))
                if self.metric == "euclidean"
                else difference.sum(axis=1)
            )
            neighbor_indexes = np.argsort(distances, kind="stable")[: self.neighbors]
            votes = Counter(self.y_[neighbor_indexes].tolist())
            predictions.append(votes.most_common(1)[0][0])
        return np.asarray(predictions)


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, iterations: int = 2_000) -> None:
        if learning_rate <= 0 or iterations < 1:
            raise ValueError("learning_rate and iterations must be positive")
        self.learning_rate = learning_rate
        self.iterations = iterations

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -500, 500)
        return 1.0 / (1.0 + np.exp(-clipped))

    def fit(self, X: object, y: object) -> LogisticRegression:
        features = as_feature_matrix(X)
        target = as_target_vector(y, len(features)).astype(float)
        if set(np.unique(target)) - {0.0, 1.0}:
            raise ValueError("binary logistic regression requires labels 0 and 1")

        self.weights_ = np.zeros(features.shape[1], dtype=float)
        self.bias_ = 0.0
        sample_count = len(features)
        for _ in range(self.iterations):
            probabilities = self._sigmoid(features @ self.weights_ + self.bias_)
            error = probabilities - target
            self.weights_ -= self.learning_rate * (features.T @ error) / sample_count
            self.bias_ -= self.learning_rate * float(error.mean())
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        require_fitted(self, "weights_")
        features = as_feature_matrix(X)
        return self._sigmoid(features @ self.weights_ + self.bias_)

    def predict(self, X: object) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


class MulticlassPerceptron:
    def __init__(self, learning_rate: float = 1.0, epochs: int = 25) -> None:
        if learning_rate <= 0 or epochs < 1:
            raise ValueError("learning_rate and epochs must be positive")
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X: object, y: object) -> MulticlassPerceptron:
        features = as_feature_matrix(X)
        target = as_target_vector(y, len(features))
        self.classes_ = np.unique(target)
        self.weights_ = np.zeros((len(self.classes_), features.shape[1]), dtype=float)
        self.bias_ = np.zeros(len(self.classes_), dtype=float)

        class_indexes = {label: index for index, label in enumerate(self.classes_)}
        for _ in range(self.epochs):
            for row, label in zip(features, target, strict=True):
                actual = class_indexes[label]
                predicted = int(np.argmax(self.weights_ @ row + self.bias_))
                if predicted != actual:
                    step = self.learning_rate * row
                    self.weights_[actual] += step
                    self.weights_[predicted] -= step
                    self.bias_[actual] += self.learning_rate
                    self.bias_[predicted] -= self.learning_rate
        return self

    def predict(self, X: object) -> np.ndarray:
        require_fitted(self, "weights_")
        features = as_feature_matrix(X)
        indexes = np.argmax(features @ self.weights_.T + self.bias_, axis=1)
        return self.classes_[indexes]

