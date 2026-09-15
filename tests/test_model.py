import pandas as pd

from src.train import load_data


def test_data_exists():

    X, y = load_data()

    assert X.shape[0] > 0
    assert y.shape[0] > 0


def test_required_columns():

    X, y = load_data()

    required_columns = [
        "age",
        "income",
        "tenure",
        "monthly_charges",
        "support_calls"
    ]

    for column in required_columns:
        assert column in X.columns


def test_target_values():

    X, y = load_data()

    assert set(y.unique()).issubset({0, 1})