import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../beam_search_summary_test.csv")

# Melt all sparsity-suffixed columns
id_vars = ['dataset', 'beam_width', 'k']

df_long = pd.wide_to_long(
    df,
    stubnames=['avg_edges', 'train_relevant', 'train_seen', 'train_expanded'],
    i=id_vars,
    j='coverage',
    sep='_',
    suffix=r'[\d.]+'
).reset_index()

df_long['coverage'] = df_long['coverage'].astype(float)

df_long['recall'] = df_long['train_relevant'] / df_long['k']
baseline = df_long[df_long['coverage'] == 1.0].copy()
novel    = df_long[df_long['coverage'] <  1.0].copy()
baseline_ref = baseline[['dataset', 'beam_width', 'k', 'recall', 'train_seen', 'avg_edges']] \
    .rename(columns={'recall': 'baseline_recall', 'train_seen': 'baseline_seen', 'avg_edges': 'baseline_avg_edges'})


_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', 'p']
_COLORS  = plt.cm.tab10.colors


def top_k_bw_coverage(df_novel, top_k=5, group_by=('dataset', 'k'), score_col='recall'):
    """
    Return the top-k (beam_width, coverage) combinations per group, ranked by score_col descending.
    """
    return (
        df_novel
        .sort_values(score_col, ascending=False)
        .groupby(list(group_by), sort=False)
        .head(top_k)
        [list(group_by) + ['beam_width', 'coverage', score_col]]
        .reset_index(drop=True)
    )


def plot_top_k_settings(df_novel, df_baseline, top_k=5, score_col='recall', save=False):
    """
    For each dataset: one figure with rows=recall@k values, cols=[distance comps, avg out-edges].
    Top-k novel (bw, coverage) combos shown with distinct marker+color; baseline as a dashed line.
    """
    datasets = sorted(df_novel['dataset'].unique())
    k_values = sorted(df_novel['k'].unique())
    n_k      = len(k_values)

    for dataset in datasets:
        nov_ds  = df_novel[df_novel['dataset'] == dataset]
        base_ds = df_baseline[df_baseline['dataset'] == dataset]

        # Union of top-k combos across all k values for this dataset
        top_per_k    = top_k_bw_coverage(nov_ds, top_k=top_k, group_by=('k',), score_col=score_col)
        unique_combos = (
            top_per_k[['beam_width', 'coverage']]
            .drop_duplicates()
            .sort_values(['beam_width', 'coverage'])
            .reset_index(drop=True)
        )

        fig, axes = plt.subplots(n_k, 2, figsize=(14, 5 * n_k), squeeze=False)
        fig.suptitle(f'Dataset: {dataset}', fontsize=14, fontweight='bold')

        for row_idx, k in enumerate(k_values):
            nov_k  = nov_ds[nov_ds['k'] == k]
            base_k = base_ds[base_ds['k'] == k]

            ax_dist  = axes[row_idx, 0]
            ax_edges = axes[row_idx, 1]

            for ax, xcol, xlabel in [
                (ax_dist,  'train_seen', 'Distance Computations'),
                (ax_edges, 'avg_edges',  'Avg. Out-Edges'),
            ]:
                # Baseline: line connecting bw values at coverage=1.0
                base_sorted = base_k.sort_values(xcol)
                ax.plot(
                    base_sorted[xcol], base_sorted['recall'],
                    color='black', linestyle='--', linewidth=1.5,
                    marker='x', markersize=7, zorder=3, label='Baseline (cov=1.0, bw varies)',
                )

                # Novel top-k combos
                for i, (_, combo) in enumerate(unique_combos.iterrows()):
                    bw, cov = combo['beam_width'], combo['coverage']
                    row = nov_k[(nov_k['beam_width'] == bw) & (nov_k['coverage'] == cov)]
                    if row.empty:
                        continue
                    ax.scatter(
                        row[xcol].values[0], row['recall'].values[0],
                        marker=_MARKERS[i % len(_MARKERS)],
                        color=_COLORS[i % len(_COLORS)],
                        s=90, zorder=5, label=f'bw={int(bw)}, cov={cov:.2f}',
                    )

                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_ylabel(f'Recall@{k}', fontsize=10)
                ax.set_title(f'Recall@{k} vs {xlabel}', fontsize=11)
                ax.legend(
                    loc='upper center', bbox_to_anchor=(0.5, -0.22),
                    ncol=3, fontsize=8, frameon=True,
                )

        plt.tight_layout()
        if save:
            fig.savefig(f'{dataset}_top{top_k}.png', bbox_inches='tight', dpi=150)
        plt.show()
