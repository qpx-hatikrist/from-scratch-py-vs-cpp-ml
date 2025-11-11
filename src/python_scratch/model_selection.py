import random
import math
import pandas as pd

def scr_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    shuffle: bool = True,
    random_state: int | None = None):
    n_samples = len(X)

    if isinstance(test_size, float):
        n_test = math.ceil(n_samples * test_size)
    else:
        n_test = int(test_size)

    n_test = max(1, min(n_test, n_samples - 1))

    indices = list(range(n_samples))
    if shuffle:
        rng = random.Random(random_state)
        rng.shuffle(indices)

    test_idx = indices[-n_test:]
    train_idx = indices[:-n_test]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    return X_train, X_test, y_train, y_test