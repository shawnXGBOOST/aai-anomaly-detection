"""A small dense autoencoder in PyTorch trained on normal behaviour only."""

from __future__ import annotations

import numpy as np

from aai.models.common import keep_columns


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "PyTorch is required for the autoencoder: pip install -e '.[deep]'"
        ) from exc
    return torch


class AutoencoderModel:
    """Reconstruction-error anomaly detector.

    The network compresses the standardized feature vector through a narrow bottleneck and tries
    to rebuild it. Trained on normal data only, it rebuilds normal rows well and unusual rows
    badly. The anomaly score is the mean squared reconstruction error over features, and the
    per-feature squared errors (:meth:`feature_errors`) say which features were not explained.

    Standardization uses the fit data only. Features that are constant in the fit data get a
    standard-deviation floor, so any later change in them produces a large error, which is the
    intended behaviour for flat channels.
    """

    name = "autoencoder"

    def __init__(
        self,
        hidden: int = 16,
        bottleneck: int = 4,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 64,
        std_floor: float = 1e-3,
        seed: int = 0,
        drop_constant: bool = False,
    ) -> None:
        self.hidden, self.bottleneck, self.epochs = hidden, bottleneck, epochs
        self.lr, self.batch_size, self.std_floor, self.seed = lr, batch_size, std_floor, seed
        self.drop_constant = drop_constant
        self._keep: np.ndarray | None = None
        self._net = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> AutoencoderModel:
        torch = _torch()
        X = np.asarray(X, dtype=float)
        self._keep = keep_columns(X, self.drop_constant)
        X = X[:, self._keep]
        self._mean = X.mean(axis=0)
        self._std = np.maximum(X.std(axis=0), self.std_floor)
        data = torch.tensor((X - self._mean) / self._std, dtype=torch.float32)
        d = X.shape[1]

        torch.manual_seed(self.seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(d, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, self.bottleneck),
            torch.nn.ReLU(),
            torch.nn.Linear(self.bottleneck, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, d),
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            order = torch.randperm(len(data), generator=generator)
            for start in range(0, len(data), self.batch_size):
                batch = data[order[start : start + self.batch_size]]
                loss = ((net(batch) - batch) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        net.eval()
        self._net = net
        return self

    def feature_errors(self, X: np.ndarray) -> np.ndarray:
        """Squared reconstruction error per row and feature, in standardized units."""
        if self._net is None or self._mean is None or self._std is None:
            raise RuntimeError("model is not fitted")
        torch = _torch()
        scaled = (np.asarray(X, dtype=float)[:, self._keep] - self._mean) / self._std
        data = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            errors = (self._net(data) - data) ** 2
        return errors.numpy().astype(float)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.feature_errors(X).mean(axis=1)

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Squared reconstruction error per feature, laid out on the full feature set."""
        X = np.asarray(X, dtype=float)
        out = np.zeros_like(X)
        out[:, self._keep] = self.feature_errors(X)
        return out

    def save(self, directory) -> None:
        """Write weights (``weights.pt``, tensors only) and standardization (``scaling.npz``)."""
        from pathlib import Path

        if self._net is None or self._mean is None or self._std is None or self._keep is None:
            raise RuntimeError("model is not fitted")
        torch = _torch()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), directory / "weights.pt")
        np.savez(directory / "scaling.npz", mean=self._mean, std=self._std, keep=self._keep)

    @classmethod
    def load(cls, directory, n_features: int, **hyper) -> AutoencoderModel:
        from pathlib import Path

        torch = _torch()
        directory = Path(directory)
        model = cls(**hyper)
        with np.load(directory / "scaling.npz", allow_pickle=False) as z:
            model._mean, model._std, model._keep = z["mean"], z["std"], z["keep"]
        d = len(model._keep)
        net = torch.nn.Sequential(
            torch.nn.Linear(d, model.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(model.hidden, model.bottleneck),
            torch.nn.ReLU(),
            torch.nn.Linear(model.bottleneck, model.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(model.hidden, d),
        )
        net.load_state_dict(torch.load(directory / "weights.pt", weights_only=True))
        net.eval()
        model._net = net
        if n_features < d:
            raise ValueError("saved model does not match the feature count")
        return model
