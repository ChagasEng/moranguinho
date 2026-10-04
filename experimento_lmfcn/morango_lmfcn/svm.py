"""Kernel RBF pré-computado para o SVM e seleção de vizinhos."""

import numpy as np
from sklearn.svm import SVC


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distances = (left * left).sum(1)[:, None] + (right * right).sum(1)[None, :] - 2 * left @ right.T
    return np.maximum(distances, 0.0)


def rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * squared_distances(left, right))


def fit_svm(features: np.ndarray, labels: np.ndarray, c: float):
    variance = float(features.var())
    gamma = 1.0 / (features.shape[1] * max(variance, 1e-8))
    classifier = SVC(C=c, kernel="precomputed")
    classifier.fit(rbf_kernel(features, features, gamma), labels)
    return classifier, gamma


def predict_svm(classifier, features: np.ndarray, reference: np.ndarray, gamma: float) -> np.ndarray:
    return classifier.predict(rbf_kernel(features, reference, gamma))
