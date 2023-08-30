
import os
from typing import Final

import matplotlib.pyplot as plt
import seaborn as sns

from narlsqr.analysis import MetricsAnalyzer

RESULTS_DIR: Final = 'data/results'
ANALYSIS_DIR: Final = 'data/analysis'


def random_circuits_analysis(device: str):
    os.makedirs(f'{ANALYSIS_DIR}/{device}/random', exist_ok=True)

    metrics_analyzer = MetricsAnalyzer.unpickle(f'{RESULTS_DIR}/{device}/random.pickle')
    noise_unaware = MetricsAnalyzer.unpickle(f'{RESULTS_DIR}/{device}/random_nu.pickle')
    metrics_analyzer.metrics['rl_noise_unaware'] = noise_unaware.metrics['rl']

    ax = metrics_analyzer.box_plot('added_cnot_count')

    ax.set_ybound(0)
    ax.set_xlabel('Routing Algorithm', labelpad=8.0, fontsize=14)
    ax.set_ylabel('Additional CNOTs', labelpad=8.0, fontsize=14)
    ax.get_figure().set_size_inches(8.0, 10.0)

    plt.savefig(f'{ANALYSIS_DIR}/{device}/random/added_cnots.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()

    ax = metrics_analyzer.box_plot('depth')

    ax.set_ybound(0)
    ax.set_xlabel('Routing Algorithm', labelpad=8.0, fontsize=14)
    ax.set_ylabel('Depth', labelpad=8.0, fontsize=14)
    ax.get_figure().set_size_inches(8.0, 10.0)

    plt.savefig(f'{ANALYSIS_DIR}/{device}/random/depth.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()

    ax = metrics_analyzer.box_plot('log_reliability')

    ax.set_xlabel('Routing Algorithm', labelpad=8.0, fontsize=14)
    ax.set_ylabel('Log Reliability', labelpad=8.0, fontsize=14)
    ax.get_figure().set_size_inches(8.0, 10.0)

    plt.savefig(f'{ANALYSIS_DIR}/{device}/random/log_reliability.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()


def real_circuits_analysis(device: str):
    os.makedirs(f'{ANALYSIS_DIR}/{device}/real', exist_ok=True)

    metrics_analyzer = MetricsAnalyzer.unpickle(f'{RESULTS_DIR}/{device}/real.pickle')
    noise_unaware = MetricsAnalyzer.unpickle(f'{RESULTS_DIR}/{device}/real_nu.pickle')
    metrics_analyzer.metrics['rl_noise_unaware'] = noise_unaware.metrics['rl']

    ax = metrics_analyzer.box_plot('normalized_added_cnot_count')

    ax.set_ybound(0)
    ax.set_xlabel('Routing Algorithm', labelpad=8.0, fontsize=14)
    ax.set_ylabel('Additional CNOTs (Normalized)', labelpad=8.0, fontsize=14)
    ax.get_figure().set_size_inches(8.0, 10.0)

    plt.savefig(f'{ANALYSIS_DIR}/{device}/real/added_cnots.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()

    ax = metrics_analyzer.box_plot('normalized_depth')

    ax.set_ybound(0)
    ax.set_xlabel('Routing Algorithm', labelpad=8.0, fontsize=14)
    ax.set_ylabel('Depth (Normalized)', labelpad=8.0, fontsize=14)
    ax.get_figure().set_size_inches(8.0, 10.0)

    plt.savefig(f'{ANALYSIS_DIR}/{device}/real/depth.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()

    ax = metrics_analyzer.box_plot('normalized_log_reliability')

    ax.set_xlabel('Routing Algorithm', labelpad=8.0, fontsize=14)
    ax.set_ylabel('Log Reliability (Normalized)', labelpad=8.0, fontsize=14)
    ax.get_figure().set_size_inches(8.0, 10.0)

    plt.savefig(f'{ANALYSIS_DIR}/{device}/real/log_reliability.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()


def main():
    sns.set_theme(style='whitegrid')
    devices = ['manila', 'belem', 'nairobi', 'guadalupe']

    for device in devices:
        random_circuits_analysis(device)
        real_circuits_analysis(device)


if __name__ == '__main__':
    main()
