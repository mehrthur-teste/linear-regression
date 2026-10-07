"""Carrega o modelo guardado e faz uma previsão para um valor novo."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).with_name("linear_regression.joblib")


def parse_input_value(raw_value: str) -> float:
    """Valida e converte o valor de entrada do utilizador."""
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("O valor de X deve ser um número válido.") from exc

    if not np.isfinite(value):
        raise ValueError("O valor de X deve ser finito.")

    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Faz uma previsão com o modelo treinado.")
    parser.add_argument("value", nargs="?", help="Valor de X a usar na previsão.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei {MODEL_PATH}. Executa primeiro linear_regression.py."
        )

    if args.value is None:
        raw_value = input("Introduz o valor de X: ")
    else:
        raw_value = args.value

    try:
        valor = parse_input_value(raw_value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    modelo = joblib.load(MODEL_PATH)
    previsao = modelo.predict(np.array([[valor]], dtype=float))[0]

    print(f"Previsão para X = {valor:g}: {previsao:.2f}")


if __name__ == "__main__":
    main()
