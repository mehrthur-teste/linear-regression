"""Carrega o modelo guardado e faz uma previsão para um valor novo."""

from pathlib import Path

import joblib
import numpy as np


def main() -> None:
    caminho_modelo = Path(__file__).with_name("linear_regression.joblib")
    if not caminho_modelo.exists():
        raise FileNotFoundError(
            f"Não encontrei {caminho_modelo}. Executa primeiro linear_regression.py."
        )

    modelo = joblib.load(caminho_modelo)
    valor = float(input("Introduz o valor de X: "))
    dados_novos = np.array([[valor]])
    previsao = modelo.predict(dados_novos)[0]

    print(f"Previsão para X = {valor:g}: {previsao:.2f}")


if __name__ == "__main__":
    main()
