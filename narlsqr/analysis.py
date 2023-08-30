
import os
import pickle
from collections import OrderedDict
from numbers import Real
from pathlib import Path
from typing import Self

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


class MetricsAnalyzer:
    metrics: dict[str, dict[str, list[Real]]]

    ROUTING_METHOD_NAMES = OrderedDict([
        ('rl', 'RL'),
        ('rl_noise_unaware', 'RL (Noise Unaware)'),
        ('sabre', 'SabreSwap'),
        ('stochastic', 'StochasticSwap'),
        ('basic', 'BasicSwap'),
    ])

    def __init__(self):
        self.metrics = {}

    @classmethod
    def unpickle(cls, path: str | Path) -> Self:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def pickle(self, path: str | Path):
        path = Path(path) if isinstance(path, str) else path
        os.makedirs(path.parent, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def log_metric(self, method: str, metric: str, value: Real):
        self.metrics.setdefault(method, {}).setdefault(metric, []).append(value)

    def metric_as_df(self, metric: str, *, rename_routing_methods: bool = False) -> pd.DataFrame:
        df = pd.DataFrame({
            method: method_data.get(metric, [])
            for method, method_data in self.metrics.items()
            if metric in method_data
        }).reindex(columns=MetricsAnalyzer.ROUTING_METHOD_NAMES)

        if rename_routing_methods:
            df.rename(columns=MetricsAnalyzer.ROUTING_METHOD_NAMES, inplace=True)

        return df

    def box_plot(self, metric: str, **kwargs) -> Axes:
        return sns.boxplot(self.metric_as_df(metric, rename_routing_methods=True), **kwargs)

    def violin_plot(self, metric: str, **kwargs) -> Axes:
        return sns.violinplot(self.metric_as_df(metric, rename_routing_methods=True), **kwargs)
