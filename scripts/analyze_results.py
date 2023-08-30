
import matplotlib.pyplot as plt
import seaborn as sns

from narlsqr.analysis import MetricsAnalyzer


def main():
    sns.set_theme(style='whitegrid')
    prefix = 'data/results'

    manila = MetricsAnalyzer.unpickle(f'{prefix}/manila/random.pickle')
    noise_unaware = MetricsAnalyzer.unpickle(f'{prefix}/manila/random_nu.pickle')

    manila.metrics['rl_noise_unaware'] = noise_unaware.metrics['rl']

    ax = manila.box_plot('added_cnot_count')

    # ax.set_title('IBM Manila - Number of Additional CNOTs (lower is better)')
    ax.set_ybound(0)
    ax.set_xlabel('Routing Algorithm', labelpad=8.0, fontsize=14)
    ax.set_ylabel('Number of Additional CNOTs', labelpad=8.0, fontsize=14)
    ax.get_figure().set_size_inches(7.5, 9.0)

    plt.tight_layout()
    plt.show()

    ax = manila.box_plot('depth')

    # ax.set_title('IBM Manila - Depth (lower is better)')
    ax.set_ybound(0)
    ax.set_xlabel('Routing Algorithm', labelpad=8.0, fontsize=14)
    ax.set_ylabel('Depth', labelpad=8.0, fontsize=14)
    ax.get_figure().set_size_inches(7.5, 9.0)

    plt.tight_layout()
    plt.show()

    ax = manila.box_plot('log_reliability')

    # ax.set_title('IBM Manila - Log Reliability (higher is better)')
    ax.set_xlabel('Routing Algorithm', labelpad=8.0, fontsize=14)
    ax.set_ylabel('Log Reliability', labelpad=8.0, fontsize=14)
    ax.get_figure().set_size_inches(7.5, 8.5)

    plt.tight_layout()
    plt.show()

    # manila.box_plot('depth')
    # plt.show()
    #
    # manila.box_plot('log_reliability')
    # plt.show()

    # plt.savefig('test.png', bbox_inches='tight')



if __name__ == '__main__':
    main()
