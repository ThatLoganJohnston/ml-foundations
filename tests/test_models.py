import numpy as np
import pytest

from ml_foundations import (
    KNNClassifier,
    LinearRegression,
    LogisticRegression,
    MulticlassPerceptron,
)


def test_knn_classifies_two_clusters() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [8, 8], [8, 9], [9, 8]])
    y = np.array(["near", "near", "near", "far", "far", "far"])
    model = KNNClassifier(neighbors=3).fit(X, y)
    assert model.predict([[0.2, 0.2], [8.5, 8.5]]).tolist() == ["near", "far"]


def test_linear_regression_recovers_known_line() -> None:
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = 3.0 * X[:, 0] - 2.0
    model = LinearRegression().fit(X, y)
    assert model.intercept_ == pytest.approx(-2.0)
    assert model.coefficients_[0] == pytest.approx(3.0)
    assert model.predict([[10.0]])[0] == pytest.approx(28.0)


def test_logistic_regression_separates_binary_data() -> None:
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(learning_rate=0.2, iterations=1_000).fit(X, y)
    assert np.array_equal(model.predict(X), y)


def test_perceptron_handles_multiple_classes() -> None:
    X = np.array([[3, 0], [4, 0], [0, 3], [0, 4], [-3, -3], [-4, -4]], dtype=float)
    y = np.array(["east", "east", "north", "north", "southwest", "southwest"])
    model = MulticlassPerceptron(epochs=30).fit(X, y)
    assert np.array_equal(model.predict(X), y)


def test_predict_before_fit_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        LinearRegression().predict([[1.0]])

