import json
from collections import defaultdict
from typing import Final

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from rich import print

from scripts.analyze_results import ANALYSIS_DIR, DEVICES, format_plot, save_current_plot

TRAINING_DATA_DIR: Final = 'data/training'
HIST_STATS: Final = {
    'episode_reward': 'Episode Reward',
    'episode_lengths': 'Episode Length',
}


def parse_csv(path: str) -> dict[str, pd.DataFrame]:
    data_frames = {}

    columns = ['training_iteration']
    columns.extend(f'hist_stats/{key}' for key in HIST_STATS)
    csv_df = pd.read_csv(path, usecols=columns)

    for key in HIST_STATS:
        plot_data = defaultdict(list)

        for _, iter_data in csv_df.iterrows():
            series = pd.Series(json.loads(iter_data[f'hist_stats/{key}']))
            describe = series.describe()

            q1, q3 = describe['25%'], describe['75%']
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            trimmed_series = series[(series >= lower_bound) & (series <= upper_bound)]

            trimmed_mean = trimmed_series.mean()
            trimmed_std = trimmed_series.std()

            plot_data['iteration'].append(iter_data['training_iteration'])
            plot_data['mean'].append(trimmed_mean)
            plot_data['std_low'].append(trimmed_mean - trimmed_std)
            plot_data['std_high'].append(trimmed_mean + trimmed_std)

        data_frames[key] = pd.DataFrame(plot_data)

    return data_frames


def main():
    data_frames: dict[str, pd.DataFrame]

    sns.set_theme(style='whitegrid')
    plt.rcParams['font.sans-serif'] = ['Nimbus Sans']

    for device in DEVICES:
        print(f'Plotting training data for [b cyan]{device}[/b cyan]')
        prefix = f'{ANALYSIS_DIR}/{device}/training'

        if device == 'guadalupe':
            data_frames = {}

            # Training data for Guadalupe is split across three runs, so we must concatenate it into a single DataFrame
            for i in range(3):
                csv_path = f'{TRAINING_DATA_DIR}/{device}/progress{i}.csv'
                partial_data_frames = parse_csv(csv_path)

                for key, df in partial_data_frames.items():
                    if key in data_frames:
                        data_frames[key] = pd.concat([data_frames[key], df])
                    else:
                        data_frames[key] = df
        else:
            csv_path = f'{TRAINING_DATA_DIR}/{device}/progress.csv'
            data_frames = parse_csv(csv_path)

        for key, y_label in HIST_STATS.items():
            df = data_frames[key]

            ax = sns.lineplot(df, x='iteration', y='mean')
            ax.fill_between(df['iteration'], df['std_low'], df['std_high'], alpha=0.5, color='skyblue')
            format_plot(ax, 'Training Iteration', y_label, label_size=28, font_size=32)
            save_current_plot(f'{prefix}/{key}.pdf')


if __name__ == '__main__':
    main()
