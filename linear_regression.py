"""Exemplo inicial de regressão linear com dados de demonstração."""

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main() -> None:
    # Substitui estes valores pelos teus dados: X são as entradas e y o valor a prever.
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    joblib.dump(modelo, "linear_regression.joblib")

    previsoes = modelo.predict(X_test)

    print(f"Coeficiente: {modelo.coef_[0]:.2f}")
    print(f"Intercepto: {modelo.intercept_:.2f}")
    print(f"Erro absoluto médio (MAE): {mean_absolute_error(y_test, previsoes):.2f}")
    print(f"Erro quadrático médio (MSE): {mean_squared_error(y_test, previsoes):.2f}")
    print(f"R²: {r2_score(y_test, previsoes):.2f}")

    valor = np.array([[11]])
    previsao = modelo.predict(valor)[0]
    print(f"Previsão para X = 11: {previsao:.2f}")


if __name__ == "__main__":
    main()
