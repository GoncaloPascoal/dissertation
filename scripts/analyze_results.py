
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
    ax.tick_params(labelsize=21)
    ax.get_yaxis().set_major_locator(MaxNLocator(nbins=16))
    ax.set_xlabel(x_label, labelpad=8.0, fontsize=25)
    ax.set_ylabel(y_label, labelpad=8.0, fontsize=25)
    ax.get_figure().set_size_inches(14.0, 14.0)


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

    plt.savefig(f'{prefix}/added_cnot_count.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()

    ax = metrics_analyzer.box_plot('depth')

    format_plot(ax, 'Routing Algorithm', 'Circuit Depth')

    plt.savefig(f'{prefix}/depth.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()

    ax = metrics_analyzer.box_plot('log_reliability')

    format_plot(ax, 'Routing Algorithm', 'Log Reliability')

    plt.savefig(f'{prefix}/log_reliability.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()


def real_circuits_analysis(device: str):
    prefix = f'{ANALYSIS_DIR}/{device}/real'
    os.makedirs(prefix, exist_ok=True)

    metrics_analyzer = MetricsAnalyzer.unpickle(f'{RESULTS_DIR}/{device}/real.pickle')
    noise_unaware = MetricsAnalyzer.unpickle(f'{RESULTS_DIR}/{device}/real_nu.pickle')
    metrics_analyzer.metrics['rl_noise_unaware'] = noise_unaware.metrics['rl']

    ax = metrics_analyzer.box_plot('normalized_added_cnot_count')

    for metric in ['normalized_added_cnot_count', 'normalized_depth', 'normalized_log_reliability']:
        df = metrics_analyzer.metric_as_df(metric).describe()
        df.to_csv(f'{prefix}/{metric.removeprefix("normalized_")}.csv', float_format='%.3f')

    ax.tick_params(labelsize=16)
    format_plot(ax, 'Routing Algorithm', 'Additional CNOT Gates (Normalized)')

    plt.savefig(f'{prefix}/added_cnot_count.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()

    ax = metrics_analyzer.box_plot('normalized_depth')

    format_plot(ax, 'Routing Algorithm', 'Depth (Normalized)')

    plt.savefig(f'{prefix}/depth.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()

    ax = metrics_analyzer.box_plot('normalized_log_reliability')

    format_plot(ax, 'Routing Algorithm', 'Log Reliability (Normalized)')

    plt.savefig(f'{prefix}/log_reliability.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()


def main():
    sns.set_theme(style='whitegrid')
    devices = ['manila', 'belem', 'nairobi', 'guadalupe']

    for device in devices:
        random_circuits_analysis(device)
        real_circuits_analysis(device)


if __name__ == '__main__':
    main()
