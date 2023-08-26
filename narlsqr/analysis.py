
import pickle
from numbers import Real
from typing import Self

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


class MetricsAnalyzer:
    metrics: dict[str, dict[str, list[Real]]]

    def __init__(self):
        self.metrics = {}

    @classmethod
    def unpickle(cls, path: str) -> Self:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def pickle(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def log_metric(self, method: str, metric: str, value: Real):
        self.metrics.setdefault(method, {}).setdefault(metric, []).append(value)

    def metric_as_df(self, metric: str) -> pd.DataFrame:
        return pd.DataFrame({
            method: method_data.get(metric, [])
            for method, method_data in self.metrics.items()
            if metric in method_data
        })

    def box_plot(self, metric: str, **kwargs) -> Axes:
        return sns.boxplot(self.metric_as_df(metric), **kwargs)

    def violin_plot(self, metric: str, **kwargs) -> Axes:
        return sns.violinplot(self.metric_as_df(metric), **kwargs)
