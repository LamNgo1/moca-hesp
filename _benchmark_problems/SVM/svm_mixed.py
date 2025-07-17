import logging
import math
import pathlib
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVR
from xgboost import XGBRegressor


def download_uci_data():
    if not pathlib.Path("_benchmark_problems/SVM/slice_localization_data.csv").exists():
        logging.info("slice_localization_data.csv not found. Downloading...")

        url = "https://archive.ics.uci.edu/static/public/206/relative+location+of+ct+slices+on+axial+axis.zip"
        logging.info(f"Downloading {url}")

        import requests
        response = requests.get(url)

        with open("_benchmark_problems/SVM/slice_localization_data.zip", "wb") as file:
            file.write(response.content)
        logging.info("Download completed.")
        import zipfile

        with zipfile.ZipFile("_benchmark_problems/SVM/slice_localization_data.zip", "r") as zip_ref:
            zip_ref.extractall("_benchmark_problems/SVM")
        # delete .zip file
        pathlib.Path("_benchmark_problems/SVM/slice_localization_data.zip").unlink()
        logging.info("Data extracted!")

def load_uci_data(
        n_features: Optional[int] = None,
):
    # taken from the BODi paper (https://arxiv.org/pdf/2303.01774.pdf)

    if not pathlib.Path("_benchmark_problems/SVM/slice_localization_data.csv").exists():
        download_uci_data()

    try:
        path = pathlib.Path("_benchmark_problems/SVM/slice_localization_data.csv").resolve()
        df = pd.read_csv(path, sep=",")
        data = df.to_numpy()

    except:
        raise FileNotFoundError(
            "The UCI slice_localization_data.csv dataset was not found. Please download it using the download_uci_data() function."
        )

    # Get the input data
    X = data[:, :-1]
    X -= X.min(axis=0)
    X = X[:, X.max(axis=0) > 1e-6]  # Throw away constant dimensions
    X = X / (X.max(axis=0) - X.min(axis=0))
    X = 2 * X - 1
    assert X.min() == -1 and X.max() == 1

    # Standardize targets
    y = data[:, -1]
    y = (y - y.mean()) / y.std()

    # Only keep 10,000 data points and n_features features
    shuffled_indices = np.random.RandomState(0).permutation(X.shape[0])[
                       :10_000
                       ]  # Use seed 0
    X, y = X[shuffled_indices], y[shuffled_indices]

    if n_features is not None:
        # Use Xgboost to figure out feature importances and keep only the most important features
        xgb = XGBRegressor(max_depth=8).fit(X, y)
        inds = (-xgb.feature_importances_).argsort()
        X = X[:, inds[:n_features]]

    # Train/Test split on a subset of the data
    train_n = int(math.floor(0.50 * X.shape[0]))
    train_x, train_y = X[:train_n], y[:train_n]
    test_x, test_y = X[train_n:], y[train_n:]

    return train_x, train_y, test_x, test_y


class BaseSVMMixed():
    """
    The mixed variable-type SVM benchmark
    """
    def __init__(self, n_features=50, shifted=False, **kwargs):
        super(BaseSVMMixed, self).__init__(**kwargs)
        self.n_vertices = np.array([2] * n_features)
        # Customized params
        self.name = f'svm53'
        self.categorical_idx_m = list(range(n_features))
        self.discrete_idx_m = []
        self.continuous_idx_m = [n_features, n_features+1, n_features+2]

        self.discrete_dim_m = len(self.discrete_idx_m)
        self.categorical_dim_m = len(self.categorical_idx_m)
        self.continuous_dim_m = len(self.continuous_idx_m)

        self.bounds_m = np.vstack(
            (
                np.array([[0, 1] for _ in range(self.categorical_dim_m)]),
                np.array([[0, 1] for _ in range(self.continuous_dim_m)]),
            )
        )
        self.categorical_vertices_m = [self.bounds_m[cat_idx][1] - self.bounds_m[cat_idx][0] + 1 
                                        for cat_idx in self.categorical_idx_m]

        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        self.n_features = n_features
        self.train_x, self.train_y, self.test_x, self.test_y = load_uci_data(n_features=n_features)
        self.shifted = shifted
        if self.shifted:
            self.offset_cat = torch.tensor(np.random.RandomState(2024).choice([0, 1], n_features))
            self.offset_cont = np.random.RandomState(2024).uniform(-1, 1, size=self.continuous_dim_m)
            self.name = f'shifted-{self.name}'
            print(f'offset={np.around(self.offset_cat,0).tolist() + np.around(self.offset_cont,4).tolist()}')
        else:
            self.offset_cat = np.zeros(self.categorical_dim_m)
            self.offset_cont = np.zeros(self.continuous_dim_m)
    
    def func_core(self, x):
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                X = deepcopy(x)
            elif x.ndim == 2:
                X = x.flatten()
            else: 
                raise NotImplementedError()
        elif isinstance(x, list):  
            X = np.array(x) 
        elif isinstance(x, torch.Tensor):
            X = x.detach().clone()
        else:
            raise NotImplementedError() 
        if isinstance(X, dict):
            X = np.array([val for val in X.values()])
        if not isinstance(X, torch.Tensor):
            try:
                X = torch.tensor(X)
            except:
                raise Exception('Unable to convert x to a pytorch tensor!')
        return self.evaluate(X)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim in [0, 1]:
            x = torch.unsqueeze(x, 0)
        assert x.ndim == 2
        assert x.shape[1] == self.n_features + 3

        rmses = []
        
        for _x in x:
            assert (len(_x) == self.n_features + 3), f"Expected {self.n_features + 3} dimensions, got {len(_x)}"
            if self.shifted:
                _x[self.categorical_idx_m] = torch.abs(_x[self.categorical_idx_m] - self.offset_cat)
                _x[self.continuous_idx_m] = _x[self.continuous_idx_m] + self.offset_cont
            epsilon = 0.01 * 10 ** (2 * _x[-3])  # Default = 0.1
            C = 0.01 * 10 ** (4 * _x[-2])  # Default = 1.0
            gamma = (
                (1 / self.categorical_dim_m) * 0.1 * 10 ** (2 * _x[-1])
            )  # Default = 1.0 / self.n_features
            model = SVR(C=C.item(), epsilon=epsilon.item(), gamma=gamma.item())
            inds_selected = np.where(_x[np.arange(self.categorical_dim_m)].cpu().numpy() == 1)[0]
            if len(inds_selected) == 0:  # Silly corner case with no features
                rmses.append(1.0)
            else:
                model.fit(self.train_x[:, inds_selected], self.train_y)
                y_pred = model.predict(self.test_x[:, inds_selected])
                rmse = math.sqrt(((y_pred - self.test_y) ** 2).mean(axis=0).item())
                rmses.append(rmse)
        return np.array(rmses, dtype=float)