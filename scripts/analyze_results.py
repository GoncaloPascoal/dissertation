
import os
from typing import Final

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from narlsqr.analysis import MetricsAnalyzer

RESULTS_DIR: Final = 'data/results'
ANALYSIS_DIR: Final = 'data/analysis'


def format_plot(ax: Axes, x_label: str, y_label: str):
    ax.tick_params(labelsize=23)
    ax.get_yaxis().set_major_locator(MaxNLocator(nbins=16))
    ax.set_xlabel(x_label, labelpad=8.0, fontsize=27, fontweight='bold')
    ax.set_ylabel(y_label, labelpad=8.0, fontsize=27, fontweight='bold')
    ax.get_figure().set_size_inches(14.0, 14.0)

def save_current_plot(path: str):
    plt.savefig(path, bbox_inches='tight', pad_inches=0.2)
    plt.close()


def random_circuits_analysis(device: str):
    prefix = f'{ANALYSIS_DIR}/{device}/random'
    os.makedirs(prefix, exist_ok=True)

    metrics_analyzer = MetricsAnalyzer.unpickle(f'{RESULTS_DIR}/{device}/random.pickle')
    noise_unaware = MetricsAnalyzer.unpickle(f'{RESULTS_DIR}/{device}/random_nu.pickle')
    metrics_analyzer.metrics['rl_noise_unaware'] = noise_unaware.metrics['rl']

    for metric in ['added_cnot_count', 'depth', 'log_reliability']:
        df = metrics_analyzer.metric_as_df(metric).describe()
        df.to_csv(f'{prefix}/{metric}.csv', float_format='%.3f')

    ax = metrics_analyzer.box_plot('added_cnot_count')
    format_plot(ax, 'Routing Algorithm', 'Additional CNOT Gates')
    save_current_plot(f'{prefix}/added_cnot_count.pdf')

    ax = metrics_analyzer.box_plot('depth')
    format_plot(ax, 'Routing Algorithm', 'Circuit Depth')
    save_current_plot(f'{prefix}/depth.pdf')

    ax = metrics_analyzer.box_plot('log_reliability')
    format_plot(ax, 'Routing Algorithm', 'Log Reliability')
    save_current_plot(f'{prefix}/log_reliability.pdf')


def real_circuits_analysis(device: str):
    prefix = f'{ANALYSIS_DIR}/{device}/real'
    os.makedirs(prefix, exist_ok=True)

    metrics_analyzer = MetricsAnalyzer.unpickle(f'{RESULTS_DIR}/{device}/real.pickle')
    noise_unaware = MetricsAnalyzer.unpickle(f'{RESULTS_DIR}/{device}/real_nu.pickle')
    metrics_analyzer.metrics['rl_noise_unaware'] = noise_unaware.metrics['rl']

    for metric in ['normalized_added_cnot_count', 'normalized_depth', 'normalized_log_reliability']:
        df = metrics_analyzer.metric_as_df(metric).describe()
        df.to_csv(f'{prefix}/{metric.removeprefix("normalized_")}.csv', float_format='%.3f')

    ax = metrics_analyzer.box_plot('normalized_added_cnot_count')
    ax.tick_params(labelsize=16)
    format_plot(ax, 'Routing Algorithm', 'Additional CNOT Gates (Normalized)')
    save_current_plot(f'{prefix}/added_cnot_count.pdf')

    ax = metrics_analyzer.box_plot('normalized_depth')
    format_plot(ax, 'Routing Algorithm', 'Depth (Normalized)')
    save_current_plot(f'{prefix}/depth.pdf')

    ax = metrics_analyzer.box_plot('normalized_log_reliability')
    format_plot(ax, 'Routing Algorithm', 'Log Reliability (Normalized)')
    save_current_plot(f'{prefix}/log_reliability.pdf')


def main():
    sns.set_theme(style='whitegrid')
    plt.rcParams['font.sans-serif'] = ['Nimbus Sans']

    devices = ['manila', 'belem', 'nairobi', 'guadalupe', 'mumbai']

    for device in devices:
        random_circuits_analysis(device)
        real_circuits_analysis(device)


if __name__ == '__main__':
    main()
