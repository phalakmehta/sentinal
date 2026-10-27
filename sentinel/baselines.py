"""Baselines from the permitted list, run causally on the same stream.

Every baseline produces a per-point score using only past data (models are
fit on the warm-up segment, then applied to each trailing window). Scores are
standardised by the warm-up score distribution so one global threshold `k`
applies to every series; alarms then go through the same event logic as
Sentinel (cooldown, capped event length).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import warnings

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM


def _lagged(values, w):
    pad = np.concatenate([np.full(w - 1, values[0]), values])
    return sliding_window_view(pad, w)          # row i = values[i-w+1 .. i]


def rolling_z(values, warmup, window=100):
    s = pd.Series(values)
    mu = s.rolling(window, min_periods=10).mean().shift(1)
    sd = s.rolling(window, min_periods=10).std().shift(1)
    return np.nan_to_num(((s - mu).abs() / (sd + 1e-9)).to_numpy())


def _model_scores(model, values, warmup, w):
    X = _lagged(values, w)
    X = (X - X[:warmup].mean()) / (X[:warmup].std() + 1e-9)
    model.fit(X[:warmup])
    return -model.score_samples(X)


def isolation_forest(values, warmup, w=16):
    return _model_scores(IsolationForest(n_estimators=100, random_state=0), values, warmup, w)


def lof(values, warmup, w=16):
    return _model_scores(LocalOutlierFactor(n_neighbors=20, novelty=True), values, warmup, w)


def ocsvm(values, warmup, w=16):
    return _model_scores(OneClassSVM(nu=0.01, gamma="scale"), values, warmup, w)


def _windows(values, warmup, w):
    X = _lagged(values, w)
    return (X - X[:warmup].mean()) / (X[:warmup].std() + 1e-9)


def autoencoder(values, warmup, w=16):
    """Dense autoencoder 16 -> 8 -> 4 -> 8 -> 16 trained to rebuild warm-up windows;
    score = reconstruction error of each trailing window."""
    X = _windows(values, warmup, w)
    ae = MLPRegressor(hidden_layer_sizes=(8, 4, 8), activation="tanh", max_iter=2000, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ae.fit(X[:warmup], X[:warmup])
    return ((ae.predict(X) - X) ** 2).mean(axis=1)


def temporal_clustering(values, warmup, w=16, k=8):
    """k-means on warm-up windows; score = distance of each trailing window to its nearest centre."""
    X = _windows(values, warmup, w)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X[:warmup])
    return km.transform(X).min(axis=1)


BASELINES = {"rolling_z": rolling_z, "isolation_forest": isolation_forest,
             "lof": lof, "ocsvm": ocsvm, "autoencoder": autoencoder,
             "temporal_clustering": temporal_clustering}


def standardise(scores, warmup):
    ref = scores[:warmup]
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) + 1e-9
    z = (scores - med) / mad
    z[:warmup] = 0.0
    return z


def alarms_from_scores(z, k, cooldown=300, max_len=30):
    labels = np.zeros(len(z), dtype=bool)
    last, run = -10 ** 9, 0
    on = False
    for i, v in enumerate(z):
        if on:
            run += 1
            if v < k or run >= max_len:
                on = False
        elif v >= k and i - last > cooldown:
            on, run, last = True, 0, i
        labels[i] = on
    return labels
