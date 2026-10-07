"""Treina um modelo de regressão linear simples sobre dados de exemplo."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).with_name("linear_regression.joblib")


def build_sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Cria um conjunto de dados simples para demonstrar o uso do modelo."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0])
    return X, y


def validate_features_and_target(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Valida o formato dos dados de treino."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("X deve ser um array 2D com a forma (n_amostras, 1).")
    if y.ndim != 1:
        raise ValueError("y deve ser um array 1D.")
    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo número de linhas.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X e y não podem conter valores vazios, infinitos ou NaN.")

    return X, y


def train_model() -> LinearRegression:
    """Treina e guarda o modelo de regressão linear."""
    X, y = build_sample_data()
    X, y = validate_features_and_target(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

    predictions = model.predict(X_test)
    print(f"Coeficiente: {model.coef_[0]:.2f}")
    print(f"Intercepto: {model.intercept_:.2f}")
    print(f"Erro absoluto médio (MAE): {mean_absolute_error(y_test, predictions):.2f}")
    print(f"Erro quadrático médio (MSE): {mean_squared_error(y_test, predictions):.2f}")
    print(f"R²: {r2_score(y_test, predictions):.2f}")

    return model


def predict_value(model: LinearRegression, value: float) -> float:
    """Faz uma previsão para um único valor de entrada."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("O valor de entrada deve ser um número finito.")
    return float(model.predict(np.array([[value]], dtype=float))[0])


def main() -> None:
    """Ponto de entrada do script de treino."""
    model = train_model()
    prediction = predict_value(model, 11.0)
    print(f"Previsão para X = 11: {prediction:.2f}")


if __name__ == "__main__":
    main()
