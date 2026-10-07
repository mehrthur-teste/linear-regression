import numpy as np
import pytest

from linear_regression import build_sample_data, validate_features_and_target
from predict import parse_input_value


def test_build_sample_data_returns_valid_arrays() -> None:
    X, y = build_sample_data()

    assert X.ndim == 2
    assert X.shape[1] == 1
    assert y.ndim == 1
    assert len(X) == len(y)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


def test_validate_features_and_target_rejects_invalid_shapes() -> None:
    X = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="forma"):
        validate_features_and_target(X, y)


def test_parse_input_value_rejects_invalid_numbers() -> None:
    with pytest.raises(ValueError, match="número válido"):
        parse_input_value("abc")

    with pytest.raises(ValueError, match="finito"):
        parse_input_value("nan")
