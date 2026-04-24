"""
mlc_plots.py
============
Tái hiện các figures trong Section 9 của bài báo.

Figures được tái hiện:
    - Figure 1: Hamming loss vs cost of abstention (giống Figure 1 paper)
    - Figure 2: Rank loss vs cost
    - Figure 3: Subset 0/1 loss vs cost
    - Figure 4: F1-measure vs cost
    - Figure 5: Jaccard measure vs cost
    - Figure 7: Average gain vs MLC performance (meta-analysis)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
import os


# ============================================================
# STYLE — giống paper
# ============================================================

# Style tương tự paper: ABS=dotted, MLC=solid, PAR=dashed cyan, SEP=solid blue
STYLES = {
    'MLC':  {'color': 'black',       'linestyle': '-',  'linewidth': 1.5, 'label': 'MLC'},
    'ABS':  {'color': 'gray',        'linestyle': '--', 'linewidth': 1.0, 'label': 'ABS'},
    'SEP':  {'color': '#1f77b4',     'linestyle': '-',  'linewidth': 1.5, 'label': 'SEP'},
    'PAR':  {'color': '#d62728',     'linestyle': '--', 'linewidth': 1.5, 'label': 'PAR'},
}

LOSS_LABELS = {
    'hamming':  'Hamming loss (%)',
    'subset01': 'Subset 0/1 loss (%)',
    'fmeasure': 'F-measure (%)',
    'jaccard':  'Jaccard measure (%)',
}

LOSS_SCALE = {
    # Nhân với 100 để hiển thị %, tương tự paper
    'hamming':  100.0,
    'subset01': 100.0,
    'fmeasure': 100.0,  # paper hiển thị accuracy (1-loss)*100
    'jaccard':  100.0,
}


def plot_loss_vs_cost(
    results: Dict,
    dataset_names: List[str],
    loss_name: str,
    classifier_name: str,
    output_path: str = 'outputs/',
    figsize: Optional[Tuple] = None
) -> str:
    """
    Tái hiện Figure 1-5 trong paper.

    Layout: N_datasets cột, mỗi cột có 2 subplots:
        - Trái: loss vs cost of abstention
        - Phải: abstention size vs cost of abstention

    Paper style: x-axis = cost index 1..10 (cn * scale)
    """
    n_ds = len(dataset_names)
    if figsize is None:
        figsize = (4 * n_ds, 6)

    fig, axes = plt.subplots(2, n_ds, figsize=figsize)
    if n_ds == 1:
        axes = axes.reshape(2, 1)

    is_accuracy = loss_name in ['fmeasure', 'jaccard']
    scale = LOSS_SCALE[loss_name]

    for col, ds_name in enumerate(dataset_names):
        ax_loss = axes[0, col]
        ax_abs = axes[1, col]

        if ds_name not in results or loss_name not in results[ds_name]:
            ax_loss.set_visible(False)
            ax_abs.set_visible(False)
            continue

        ds_results = results[ds_name][loss_name]

        if classifier_name not in ds_results:
            ax_loss.text(0.5, 0.5, 'No data', ha='center', va='center',
                         transform=ax_loss.transAxes, fontsize=8)
            continue

        df = ds_results[classifier_name]
        x = df['cost_pct'].values

        # --- Loss panel ---
        ax_loss.axhline(
            df['loss_mlc'].mean() * scale,
            color=STYLES['MLC']['color'],
            linestyle=STYLES['MLC']['linestyle'],
            linewidth=STYLES['MLC']['linewidth'],
            label='MLC'
        )
        ax_loss.axhline(
            df['loss_abs'].mean() * scale,
            color=STYLES['ABS']['color'],
            linestyle=STYLES['ABS']['linestyle'],
            linewidth=STYLES['ABS']['linewidth'],
            label='ABS'
        )

        if is_accuracy:
            # F-measure và Jaccard: hiển thị accuracy = (1-loss)*100
            ax_loss.plot(x, (1 - df['loss_sep']) * scale, **{k: v for k, v in STYLES['SEP'].items() if k != 'label'}, label='SEP')
            ax_loss.plot(x, (1 - df['loss_par']) * scale, **{k: v for k, v in STYLES['PAR'].items() if k != 'label'}, label='PAR')
            ax_loss.set_ylabel(LOSS_LABELS[loss_name], fontsize=7)
        else:
            ax_loss.plot(x, df['loss_sep'] * scale, **{k: v for k, v in STYLES['SEP'].items() if k != 'label'}, label='SEP')
            ax_loss.plot(x, df['loss_par'] * scale, **{k: v for k, v in STYLES['PAR'].items() if k != 'label'}, label='PAR')
            ax_loss.set_ylabel(LOSS_LABELS.get(loss_name, 'Loss (%)'), fontsize=7)

        ax_loss.set_title(f'({chr(ord("a") + col)}) {ds_name}', fontsize=8)
        ax_loss.tick_params(labelsize=6)
        ax_loss.set_xlabel('Cost of abstention', fontsize=6)

        # --- Abstention size panel ---
        ax_abs.plot(x, df['abs_size_sep'] * 100, **{k: v for k, v in STYLES['SEP'].items() if k != 'label'}, label='SEP')
        ax_abs.plot(x, df['abs_size_par'] * 100, **{k: v for k, v in STYLES['PAR'].items() if k != 'label'}, label='PAR')
        ax_abs.set_ylabel('Abstention size (%)', fontsize=7)
        ax_abs.set_xlabel('Cost of abstention', fontsize=6)
        ax_abs.tick_params(labelsize=6)
        ax_abs.set_ylim(0, 105)

    # Legend chung
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    loss_title = {
        'hamming': 'Hamming Loss',
        'subset01': 'Subset 0/1 Loss',
        'fmeasure': 'F1-Measure',
        'jaccard': 'Jaccard Measure',
    }.get(loss_name, loss_name)

    fig.suptitle(
        f'{loss_title} — {classifier_name.upper()} '
        f'(reproducing paper Figure)',
        fontsize=10, y=1.01
    )

    plt.tight_layout()

    fname = f'{output_path}fig_{loss_name}_{classifier_name}.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")
    return fname


def plot_meta_analysis(
    all_results: Dict,
    loss_name: str,
    output_path: str = 'outputs/'
) -> str:
    """
    Tái hiện Figure 7: Average gain vs MLC performance.

    Quan sát quan trọng từ paper:
    "Khi MLC loss càng cao (classifier yếu), gain từ abstention càng lớn"
    Điều này xác nhận: reliable classifier biết khi nào nên abstain.
    """
    from mlc_experiments import compute_average_gain

    gains_sep, gains_par = [], []
    abs_sep, abs_par = [], []
    mlc_losses = []

    for ds_name, ds_results in all_results.items():
        if loss_name not in ds_results:
            continue
        for clf_name, df in ds_results[loss_name].items():
            if df is None or len(df) == 0:
                continue
            metrics = compute_average_gain(df)
            gains_sep.append(metrics['gain_sep'])
            gains_par.append(metrics['gain_par'])
            abs_sep.append(metrics['avg_abs_size_sep'])
            abs_par.append(metrics['avg_abs_size_par'])
            mlc_losses.append(metrics['avg_loss_mlc'])

    if not gains_sep:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Average gain vs MLC loss
    ax1.scatter(mlc_losses, gains_sep, label='SEP', marker='o',
                color=STYLES['SEP']['color'], alpha=0.7, s=60)
    ax1.scatter(mlc_losses, gains_par, label='PAR', marker='x',
                color=STYLES['PAR']['color'], alpha=0.7, s=60)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_xlabel(f'MLC {loss_name} loss (classifier baseline)', fontsize=9)
    ax1.set_ylabel('Average gain (MLC loss - abstention loss)', fontsize=9)
    ax1.set_title('(a) Average gain', fontsize=10)
    ax1.legend(fontsize=8)

    # Right: Average abstention size vs MLC loss
    ax2.scatter(mlc_losses, [a * 100 for a in abs_sep], label='SEP',
                marker='o', color=STYLES['SEP']['color'], alpha=0.7, s=60)
    ax2.scatter(mlc_losses, [a * 100 for a in abs_par], label='PAR',
                marker='x', color=STYLES['PAR']['color'], alpha=0.7, s=60)
    ax2.set_xlabel(f'MLC {loss_name} loss', fontsize=9)
    ax2.set_ylabel('Average abstention size (%)', fontsize=9)
    ax2.set_title('(b) Average abstention size', fontsize=10)
    ax2.legend(fontsize=8)

    fig.suptitle(
        f'Figure 7 reproduction: Gain & abstention size vs classifier performance\n'
        f'Loss: {loss_name} | '
        f'{len(gains_sep)} configurations',
        fontsize=10
    )
    plt.tight_layout()

    fname = f'{output_path}fig_meta_{loss_name}.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")
    return fname


def plot_summary_table(
    all_results: Dict,
    output_path: str = 'outputs/'
) -> str:
    """
    Tạo bảng tóm tắt kết quả cho tất cả datasets và loss functions.
    """
    from mlc_experiments import compute_average_gain

    rows = []
    for ds_name, ds_data in all_results.items():
        for loss_name, loss_data in ds_data.items():
            for clf_name, df in loss_data.items():
                if df is None or len(df) == 0:
                    continue
                m = compute_average_gain(df)
                rows.append({
                    'Dataset': ds_name,
                    'Loss': loss_name,
                    'Classifier': clf_name,
                    'MLC Loss': f"{m['avg_loss_mlc']:.4f}",
                    'Gain SEP': f"{m['gain_sep']:.4f}",
                    'Gain PAR': f"{m['gain_par']:.4f}",
                    'Abs Size SEP': f"{m['avg_abs_size_sep']:.3f}",
                    'Abs Size PAR': f"{m['avg_abs_size_par']:.3f}",
                })

    if not rows:
        return None

    df_summary = pd.DataFrame(rows)
    print("\n" + "="*90)
    print("SUMMARY TABLE")
    print("="*90)
    print(df_summary.to_string(index=False))
    print("="*90)

    # Save CSV
    csv_path = f'{output_path}results_summary.csv'
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")
    return csv_path
