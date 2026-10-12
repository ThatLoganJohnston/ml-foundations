from __future__ import annotations

import numpy as np

from ._validation import as_feature_matrix, as_target_vector, require_fitted


class LinearRegression:
    def fit(self, X: object, y: object) -> LinearRegression:
        features = as_feature_matrix(X)
        target = as_target_vector(y, len(features)).astype(float)
        design = np.column_stack((np.ones(len(features)), features))
        parameters = np.linalg.pinv(design) @ target
        self.intercept_ = float(parameters[0])
        self.coefficients_ = parameters[1:]
        return self

    def predict(self, X: object) -> np.ndarray:
        require_fitted(self, "coefficients_")
        features = as_feature_matrix(X)
        return features @ self.coefficients_ + self.intercept_

