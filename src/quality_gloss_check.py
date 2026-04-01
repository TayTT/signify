#!/usr/bin/env python3
"""
Gloss distribution quality checks for Phoenix-format annotation CSV files.
"""

import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ---------- CSV loading ----------

def load_annotations(csv_path: str) -> pd.DataFrame:
    """
    Load a Phoenix-format annotations CSV.
    Tries common separators, returns df with columns: id, folder, signer, annotation.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f'annotations file not found: {path}')

    if path.suffix.lower() == '.xlsx':
        df = pd.read_excel(path)
    else:
        df = None
        for sep in ['|', ',', '\t', ';']:
            try:
                test = pd.read_csv(path, sep=sep, nrows=3)
                if len(test.columns) >= 4:
                    df = pd.read_csv(path, sep=sep)
                    break
            except Exception:
                continue
        if df is None:
            raise ValueError(f'could not parse {path} with any known separator')

    required = ['id', 'folder', 'signer', 'annotation']
    if not all(c in df.columns for c in required):
        if len(df.columns) < 4:
            raise ValueError(f'expected at least 4 columns, got {len(df.columns)}')
        df = df.rename(columns=dict(zip(df.columns[:4], required)))

    df = df[required].copy().dropna()
    df['id']         = df['id'].astype(str).str.strip()
    df['annotation'] = df['annotation'].astype(str).str.strip()
    df['signer']     = df['signer'].astype(str).str.strip()
    df = df[df['annotation'].str.len() > 0]
    return df


# ---------- gloss helpers ----------

def get_gloss_tokens(df: pd.DataFrame) -> List[str]:
    """flat list of every gloss token across all annotations"""
    tokens = []
    for ann in df['annotation']:
        tokens.extend(ann.split())
    return tokens


def get_gloss_counts(df: pd.DataFrame) -> Counter:
    return Counter(get_gloss_tokens(df))


def get_sequence_lengths(df: pd.DataFrame) -> List[int]:
    return [len(ann.split()) for ann in df['annotation']]


# ---------- shared figure helpers ----------

def _save_or_show(fig, output_path):
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'saved to {output_path}')
    else:
        plt.show()
    plt.close(fig)


# ---------- 1. gloss frequency ----------

def plot_gloss_frequency(
    csv_path:      str,
    output_path:   Optional[str] = None,
    figsize_scale: float         = 1.0,
    top_n:         Optional[int] = None,   # show only top N glosses; None = all
    min_count:     int           = 1,      # hide glosses with fewer samples
    csv_out:       Optional[str] = None,
):
    """
    Bar chart of per-gloss sample counts, sorted descending.
    Header shows vocabulary size, hapax count, and singleton rate.
    """
    df     = load_annotations(csv_path)
    counts = get_gloss_counts(df)

    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    items = [(g, c) for g, c in items if c >= min_count]
    if top_n:
        items = items[:top_n]

    glosses = [g for g, _ in items]
    freqs   = [c for _, c in items]
    n_shown = len(glosses)

    vocab_size  = len(counts)
    hapax_count = sum(1 for c in counts.values() if c == 1)
    total_tokens = sum(counts.values())

    fig_w = max(12.0, n_shown / 5 * figsize_scale)
    fig_h = max(5.0, 5.0 * figsize_scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bars = ax.bar(range(n_shown), freqs, color='#2c5f8a', zorder=2)
    ax.set_xticks(range(n_shown))
    ax.set_xticklabels(glosses, rotation=90, fontsize=max(4, 8 - n_shown // 50))
    ax.set_ylabel('sample count', fontsize=10)
    ax.set_xlabel('gloss', fontsize=10)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4, zorder=0)
    ax.set_xlim(-0.5, n_shown - 0.5)

    # mean line
    mean_count = np.mean(freqs)
    ax.axhline(mean_count, color='tomato', linewidth=1.0, linestyle='--',
               label=f'mean: {mean_count:.1f}')
    ax.legend(fontsize=9)

    lines = [
        f'samples: {len(df)}  |  vocab: {vocab_size}  |  total tokens: {total_tokens}',
        f'hapax legomena (count=1): {hapax_count} ({hapax_count/vocab_size*100:.1f}% of vocab)',
        f'showing: {n_shown} glosses' + (f' (top {top_n})' if top_n else '') +
        (f'  min_count={min_count}' if min_count > 1 else ''),
    ]
    fig.suptitle('\n'.join(lines), fontsize=9, family='monospace', y=0.99,
                 va='top', ha='center', linespacing=1.6)

    if csv_out:
        with open(csv_out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['gloss', 'count', 'pct_of_tokens'])
            for g, c in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                w.writerow([g, c, round(c / total_tokens * 100, 4)])
        print(f'saved csv to {csv_out}')

    _save_or_show(fig, output_path)


# ---------- 2. sequence length distribution ----------

def plot_sequence_lengths(
    csv_path:      str,
    output_path:   Optional[str] = None,
    figsize_scale: float         = 1.0,
    csv_out:       Optional[str] = None,
):
    """
    Histogram of gloss count per annotation sequence.
    Percentile markers help set max_sequence_length in config.
    """
    df      = load_annotations(csv_path)
    lengths = get_sequence_lengths(df)
    lengths_arr = np.array(lengths)

    pcts = {p: int(np.percentile(lengths_arr, p)) for p in [50, 75, 90, 95, 99, 100]}

    fig_w = max(10.0, 10.0 * figsize_scale)
    fig_h = max(5.0,  5.0  * figsize_scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bins = range(1, max(lengths) + 2)
    ax.hist(lengths, bins=bins, color='#2c5f8a', zorder=2, edgecolor='white', linewidth=0.3)

    colors = ['#e07b39', '#c94040', '#8b008b', '#2e8b57', '#1a1a1a']
    for (p, val), col in zip(pcts.items(), colors):
        ax.axvline(val, color=col, linewidth=1.2, linestyle='--',
                   label=f'p{p}: {val}')

    ax.set_xlabel('glosses per sequence', fontsize=10)
    ax.set_ylabel('sample count', fontsize=10)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4, zorder=0)
    ax.legend(fontsize=9, title='percentiles', title_fontsize=8)

    lines = [
        f'samples: {len(df)}  |  mean: {lengths_arr.mean():.1f}  '
        f'std: {lengths_arr.std():.1f}  min: {lengths_arr.min()}  max: {lengths_arr.max()}',
        f'p50: {pcts[50]}  p90: {pcts[90]}  p95: {pcts[95]}  p99: {pcts[99]}  '
        f'max: {pcts[100]}  <- suggested max_sequence_length values',
    ]
    fig.suptitle('\n'.join(lines), fontsize=9, family='monospace', y=0.99,
                 va='top', ha='center', linespacing=1.6)

    if csv_out:
        with open(csv_out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['length', 'count'])
            for l, c in sorted(Counter(lengths).items()):
                w.writerow([l, c])
        print(f'saved csv to {csv_out}')

    _save_or_show(fig, output_path)


# ---------- 3. rank-frequency (zipf) ----------

def plot_rank_frequency(
    csv_path:      str,
    output_path:   Optional[str] = None,
    figsize_scale: float         = 1.0,
):
    """
    Log-log plot of gloss rank vs frequency.
    A straight line indicates Zipf's law. The steeper the tail, the more
    dominated the vocabulary is by rare glosses.
    """
    df     = load_annotations(csv_path)
    counts = get_gloss_counts(df)

    sorted_counts = sorted(counts.values(), reverse=True)
    ranks  = np.arange(1, len(sorted_counts) + 1)
    freqs  = np.array(sorted_counts, dtype=float)

    # ideal zipf reference: f = f_max / rank
    zipf_ref = freqs[0] / ranks

    fig_w = max(8.0, 8.0 * figsize_scale)
    fig_h = max(5.0, 5.0 * figsize_scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.plot(ranks, freqs,    color='#2c5f8a', linewidth=1.5, label='observed')
    ax.plot(ranks, zipf_ref, color='tomato',  linewidth=1.0, linestyle='--',
            alpha=0.7, label='ideal Zipf (f∝1/rank)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('rank (log)', fontsize=10)
    ax.set_ylabel('frequency (log)', fontsize=10)
    ax.grid(which='both', linestyle='--', linewidth=0.3, alpha=0.4)
    ax.legend(fontsize=9)

    # compute deviation from zipf as a rough measure
    log_r    = np.log(ranks)
    log_f    = np.log(freqs)
    slope, _ = np.polyfit(log_r, log_f, 1)

    vocab_size   = len(counts)
    total_tokens = sum(counts.values())
    lines = [
        f'vocab: {vocab_size}  |  total tokens: {total_tokens}',
        f'log-log slope: {slope:.2f}  (ideal Zipf = -1.0)',
    ]
    fig.suptitle('\n'.join(lines), fontsize=9, family='monospace', y=0.99,
                 va='top', ha='center', linespacing=1.6)

    _save_or_show(fig, output_path)


# ---------- 4. coverage curve ----------

def plot_coverage(
    csv_path:      str,
    output_path:   Optional[str] = None,
    figsize_scale: float         = 1.0,
    csv_out:       Optional[str] = None,
):
    """
    Cumulative token coverage as glosses are added by descending frequency.
    Shows how many unique glosses are needed to cover X% of all tokens.
    Helps decide vocabulary pruning thresholds.
    """
    df     = load_annotations(csv_path)
    counts = get_gloss_counts(df)

    sorted_counts = sorted(counts.values(), reverse=True)
    total         = sum(sorted_counts)
    cumulative    = np.cumsum(sorted_counts) / total * 100
    n_glosses     = np.arange(1, len(sorted_counts) + 1)

    coverage_milestones = {}
    for pct in [50, 80, 90, 95, 99, 100]:
        idx = int(np.searchsorted(cumulative, pct))
        coverage_milestones[pct] = min(idx + 1, len(n_glosses))

    fig_w = max(9.0, 9.0 * figsize_scale)
    fig_h = max(5.0, 5.0 * figsize_scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.plot(n_glosses, cumulative, color='#2c5f8a', linewidth=1.5)
    ax.fill_between(n_glosses, cumulative, alpha=0.12, color='#2c5f8a')

    colors = ['#e07b39', '#c94040', '#8b008b', '#2e8b57', '#555555', '#1a1a1a']
    for (pct, n_needed), col in zip(coverage_milestones.items(), colors):
        ax.axvline(n_needed, color=col, linewidth=0.9, linestyle='--',
                   label=f'{pct}% @ {n_needed} glosses')
        ax.axhline(pct, color=col, linewidth=0.5, linestyle=':', alpha=0.5)

    ax.set_xlabel('number of glosses (sorted by frequency)', fontsize=10)
    ax.set_ylabel('cumulative token coverage %', fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(linestyle='--', linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=8, title='coverage milestones', title_fontsize=8, loc='lower right')

    lines = [
        f'vocab: {len(counts)}  |  total tokens: {total}',
        '  '.join(f'{p}%={n}g' for p, n in coverage_milestones.items()),
    ]
    fig.suptitle('\n'.join(lines), fontsize=9, family='monospace', y=0.99,
                 va='top', ha='center', linespacing=1.6)

    if csv_out:
        with open(csv_out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['n_glosses', 'cumulative_coverage_pct'])
            for n, c in zip(n_glosses, cumulative):
                w.writerow([int(n), round(float(c), 4)])
        print(f'saved csv to {csv_out}')

    _save_or_show(fig, output_path)


# ---------- 5. signer distribution ----------

def plot_signer_distribution(
    csv_path:      str,
    output_path:   Optional[str] = None,
    figsize_scale: float         = 1.0,
    csv_out:       Optional[str] = None,
):
    """
    Bar chart of sample count per signer.
    Signer imbalance can cause the model to overfit to dominant signers' style.
    """
    df           = load_annotations(csv_path)
    signer_counts = df['signer'].value_counts().sort_values(ascending=False)
    n_signers     = len(signer_counts)

    fig_w = max(8.0, n_signers * 0.6 * figsize_scale)
    fig_h = max(5.0, 5.0 * figsize_scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.bar(signer_counts.index, signer_counts.values, color='#4a7c59', zorder=2)
    ax.set_xlabel('signer', fontsize=10)
    ax.set_ylabel('sample count', fontsize=10)
    ax.set_xticks(range(n_signers))
    ax.set_xticklabels(signer_counts.index, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4, zorder=0)

    mean_s = signer_counts.mean()
    ax.axhline(mean_s, color='tomato', linewidth=1.0, linestyle='--',
               label=f'mean: {mean_s:.1f}')
    ax.legend(fontsize=9)

    # imbalance ratio: max / min
    imbalance = signer_counts.max() / signer_counts.min() if signer_counts.min() > 0 else float('inf')
    lines = [
        f'samples: {len(df)}  |  signers: {n_signers}  |  '
        f'imbalance ratio (max/min): {imbalance:.1f}x',
        f'most: {signer_counts.index[0]} ({signer_counts.iloc[0]})  '
        f'least: {signer_counts.index[-1]} ({signer_counts.iloc[-1]})',
    ]
    fig.suptitle('\n'.join(lines), fontsize=9, family='monospace', y=0.99,
                 va='top', ha='center', linespacing=1.6)

    if csv_out:
        with open(csv_out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['signer', 'count', 'pct'])
            for signer, count in signer_counts.items():
                w.writerow([signer, count, round(count / len(df) * 100, 2)])
        print(f'saved csv to {csv_out}')

    _save_or_show(fig, output_path)


# ---------- 6. split comparison (OOV) ----------

def plot_split_comparison(
    train_csv:     str,
    compare_csv:   str,
    output_path:   Optional[str] = None,
    figsize_scale: float         = 1.0,
    top_n:         int           = 30,   # top N oov glosses to show by name
    csv_out:       Optional[str] = None,
):
    """
    Compares gloss vocabulary between two splits (e.g. train vs dev).
    Shows OOV glosses in the comparison split not seen in train,
    shared vocab overlap, and per-gloss frequency comparison for shared glosses.
    """
    train_df  = load_annotations(train_csv)
    cmp_df    = load_annotations(compare_csv)

    train_counts = get_gloss_counts(train_df)
    cmp_counts   = get_gloss_counts(cmp_df)

    train_vocab = set(train_counts)
    cmp_vocab   = set(cmp_counts)

    oov_glosses    = cmp_vocab - train_vocab       # in compare, not in train
    shared_glosses = cmp_vocab & train_vocab
    train_only     = train_vocab - cmp_vocab       # in train, not in compare

    cmp_total_tokens = sum(cmp_counts.values())
    oov_tokens       = sum(cmp_counts[g] for g in oov_glosses)
    oov_token_rate   = oov_tokens / cmp_total_tokens * 100 if cmp_total_tokens > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(max(14.0, 14.0 * figsize_scale),
                                             max(5.0, 5.0 * figsize_scale)))

    # left: venn-style bar showing vocab composition
    ax = axes[0]
    categories = ['train only', 'shared', 'compare OOV']
    values     = [len(train_only), len(shared_glosses), len(oov_glosses)]
    colors     = ['#2c5f8a', '#4a7c59', '#c94040']
    ax.barh(categories, values, color=colors, zorder=2)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, str(v), va='center', fontsize=9)
    ax.set_xlabel('gloss count', fontsize=10)
    ax.set_title('vocabulary overlap', fontsize=10, loc='left')
    ax.grid(axis='x', linestyle='--', linewidth=0.4, alpha=0.4, zorder=0)
    ax.set_xlim(0, max(values) * 1.15)

    # right: top OOV glosses by frequency in compare split
    ax2 = axes[1]
    top_oov = sorted(oov_glosses, key=lambda g: cmp_counts[g], reverse=True)[:top_n]
    if top_oov:
        oov_freqs = [cmp_counts[g] for g in top_oov]
        ax2.bar(range(len(top_oov)), oov_freqs, color='#c94040', zorder=2)
        ax2.set_xticks(range(len(top_oov)))
        ax2.set_xticklabels(top_oov, rotation=90,
                            fontsize=max(5, 8 - len(top_oov) // 10))
        ax2.set_ylabel('occurrences in compare split', fontsize=9)
        ax2.set_title(f'top {len(top_oov)} OOV glosses (by compare frequency)', fontsize=10, loc='left')
        ax2.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4, zorder=0)
    else:
        ax2.text(0.5, 0.5, 'no OOV glosses', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12, color='gray')
        ax2.set_title('OOV glosses', fontsize=10, loc='left')

    compare_name = Path(compare_csv).stem
    train_name   = Path(train_csv).stem
    lines = [
        f'train: {train_name} ({len(train_df)} samples, vocab {len(train_vocab)})  |  '
        f'compare: {compare_name} ({len(cmp_df)} samples, vocab {len(cmp_vocab)})',
        f'OOV glosses: {len(oov_glosses)} ({len(oov_glosses)/len(cmp_vocab)*100:.1f}% of compare vocab)  |  '
        f'OOV token rate: {oov_token_rate:.2f}%  |  '
        f'shared: {len(shared_glosses)}',
    ]
    fig.suptitle('\n'.join(lines), fontsize=9, family='monospace', y=0.99,
                 va='top', ha='center', linespacing=1.6)

    if csv_out:
        with open(csv_out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['gloss', 'status', 'train_count', 'compare_count'])
            all_glosses = train_vocab | cmp_vocab
            for g in sorted(all_glosses):
                status = 'shared' if g in shared_glosses else \
                         'train_only' if g in train_only else 'oov'
                w.writerow([g, status, train_counts.get(g, 0), cmp_counts.get(g, 0)])
        print(f'saved csv to {csv_out}')

    _save_or_show(fig, output_path)


# ---------- CLI ----------

def _add_common_args(p: argparse.ArgumentParser):
    p.add_argument('--output', default=None, help='save figure to this path')
    p.add_argument('--scale', type=float, default=1.0, dest='figsize_scale',
                   help='scale factor for figure size (default: 1.0)')
    p.add_argument('--csv', default=None, dest='csv_out',
                   help='export underlying data to this csv path')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='gloss distribution quality checks for Phoenix annotation CSVs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest='command', metavar='COMMAND')
    sub.required = True

    # -- plot-gloss-frequency --
    pgf = sub.add_parser('plot-gloss-frequency',
                         help='bar chart of sample count per gloss, sorted by frequency')
    pgf.add_argument('csv_path', help='path to annotations CSV')
    pgf.add_argument('--top-n', type=int, default=None, dest='top_n',
                     help='show only top N glosses (default: all)')
    pgf.add_argument('--min-count', type=int, default=1, dest='min_count',
                     help='hide glosses with fewer than N samples (default: 1)')
    _add_common_args(pgf)

    # -- plot-sequence-lengths --
    psl = sub.add_parser('plot-sequence-lengths',
                         help='histogram of gloss count per annotation sequence')
    psl.add_argument('csv_path', help='path to annotations CSV')
    _add_common_args(psl)

    # -- plot-rank-frequency --
    prf = sub.add_parser('plot-rank-frequency',
                         help='log-log rank vs frequency (Zipf) plot')
    prf.add_argument('csv_path', help='path to annotations CSV')
    _add_common_args(prf)

    # -- plot-coverage --
    pc = sub.add_parser('plot-coverage',
                        help='cumulative token coverage curve by gloss frequency rank')
    pc.add_argument('csv_path', help='path to annotations CSV')
    _add_common_args(pc)

    # -- plot-signer-distribution --
    psd = sub.add_parser('plot-signer-distribution',
                         help='bar chart of sample count per signer')
    psd.add_argument('csv_path', help='path to annotations CSV')
    _add_common_args(psd)

    # -- plot-split-comparison --
    psc = sub.add_parser('plot-split-comparison',
                         help='OOV analysis and vocabulary overlap between two splits')
    psc.add_argument('train_csv',   help='path to train annotations CSV')
    psc.add_argument('compare_csv', help='path to dev/test annotations CSV to compare against train')
    psc.add_argument('--top-n', type=int, default=30, dest='top_n',
                     help='number of top OOV glosses to show by name (default: 30)')
    _add_common_args(psc)

    return p


_COMMAND_HANDLERS = {
    'plot-gloss-frequency':     lambda a: plot_gloss_frequency(
        a.csv_path, a.output, a.figsize_scale, a.top_n, a.min_count, a.csv_out),
    'plot-sequence-lengths':    lambda a: plot_sequence_lengths(
        a.csv_path, a.output, a.figsize_scale, a.csv_out),
    'plot-rank-frequency':      lambda a: plot_rank_frequency(
        a.csv_path, a.output, a.figsize_scale),
    'plot-coverage':            lambda a: plot_coverage(
        a.csv_path, a.output, a.figsize_scale, a.csv_out),
    'plot-signer-distribution': lambda a: plot_signer_distribution(
        a.csv_path, a.output, a.figsize_scale, a.csv_out),
    'plot-split-comparison':    lambda a: plot_split_comparison(
        a.train_csv, a.compare_csv, a.output, a.figsize_scale, a.top_n, a.csv_out),
}


if __name__ == '__main__':
    args = _build_parser().parse_args()
    _COMMAND_HANDLERS[args.command](args)