#!/usr/bin/env python3
"""
Data quality visualization for sign language JSON landmark files.
Currently supports Phoenix-format JSONs only.
"""

import csv
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from pathlib import Path
from typing import List, Tuple, Dict, Optional


COLOR_LEFT_DETECTED  = '#1a3a6b'  # deep blue
COLOR_RIGHT_DETECTED = '#8b0000'  # deep red
COLOR_MISSING        = '#d0d0d0'  # light gray

BAR_HEIGHT  = 0.35
BAR_GAP     = 0.05   # between left/right bar of same video
VIDEO_GAP   = 0.6    # vertical space between videos


# ---------- JSON parsing helpers ----------

def _is_hand_missing(hand_data) -> bool:
    if not hand_data:
        return True
    if isinstance(hand_data, dict):
        landmarks  = hand_data.get('landmarks', [])
        confidence = hand_data.get('confidence', 0)
        return not landmarks or confidence < 0.1
    if isinstance(hand_data, list):
        return len(hand_data) == 0
    return True


def load_hand_detection(json_path: Path) -> Tuple[List[bool], List[bool]]:
    """
    Returns (left_detected, right_detected) bool lists, one entry per frame.
    True = hand was detected in that frame.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames_data = data.get('frames', {})
    if not frames_data:
        raise ValueError(f'no frames in {json_path.name}')

    sorted_frames = sorted(frames_data.items(), key=lambda x: int(x[0]))

    left_detected  = []
    right_detected = []

    for _, frame_data in sorted_frames:
        missing = frame_data.get('missing_data', {})
        if missing:
            left_missing  = missing.get('left_hand_missing', False)
            right_missing = missing.get('right_hand_missing', False)
        else:
            hands = frame_data.get('hands', {})
            left_missing  = _is_hand_missing(hands.get('left_hand', {}))
            right_missing = _is_hand_missing(hands.get('right_hand', {}))

        left_detected.append(not left_missing)
        right_detected.append(not right_missing)

    return left_detected, right_detected


# ---------- stats helpers ----------

def _get_interior_range(left_det: List[bool], right_det: List[bool]) -> Tuple[int, int]:
    """first/last frame where at least one hand is present"""
    n = len(left_det)
    start, end = 0, n - 1
    for i in range(n):
        if left_det[i] or right_det[i]:
            start = i
            break
    for i in range(n - 1, -1, -1):
        if left_det[i] or right_det[i]:
            end = i
            break
    return start, end


def _find_gaps(detected: List[bool], interior_start: int, interior_end: int) -> List[int]:
    """lengths of contiguous missing runs within the interior range"""
    gaps = []
    run = 0
    for i in range(interior_start, interior_end + 1):
        if not detected[i]:
            run += 1
        else:
            if run > 0:
                gaps.append(run)
            run = 0
    if run > 0:
        gaps.append(run)
    return gaps


def compute_stats(records: List[Dict]) -> Dict:
    """
    records: list of dicts with keys left_detected, right_detected (bool lists)
    returns summary stats dict
    """
    left_gap_lengths       = []
    right_gap_lengths      = []
    left_gaps_per_vid      = []
    right_gaps_per_vid     = []
    left_missing_interior  = []
    right_missing_interior = []
    vids_with_left_gap     = 0
    vids_with_right_gap    = 0

    for r in records:
        ld, rd = r['left_detected'], r['right_detected']
        s, e = _get_interior_range(ld, rd)

        l_gaps = _find_gaps(ld, s, e)
        r_gaps = _find_gaps(rd, s, e)

        left_gap_lengths.extend(l_gaps)
        right_gap_lengths.extend(r_gaps)
        left_gaps_per_vid.append(len(l_gaps))
        right_gaps_per_vid.append(len(r_gaps))
        left_missing_interior.append(sum(l_gaps))
        right_missing_interior.append(sum(r_gaps))

        if l_gaps:
            vids_with_left_gap += 1
        if r_gaps:
            vids_with_right_gap += 1

    n = len(records)

    def _safe(fn, lst, fallback=0):
        return fn(lst) if lst else fallback

    return {
        'n_videos': n,
        'left': {
            'max_gap':           _safe(max, left_gap_lengths),
            'avg_missing':       _safe(np.mean, left_missing_interior),
            'avg_gaps':          _safe(np.mean, left_gaps_per_vid),
            'pct_vids_affected': vids_with_left_gap / n * 100 if n else 0,
        },
        'right': {
            'max_gap':           _safe(max, right_gap_lengths),
            'avg_missing':       _safe(np.mean, right_missing_interior),
            'avg_gaps':          _safe(np.mean, right_gaps_per_vid),
            'pct_vids_affected': vids_with_right_gap / n * 100 if n else 0,
        },
    }


def _format_stats_title(stats: Dict, hands: str) -> str:
    l = stats['left']
    r = stats['right']
    lines = [f"videos: {stats['n_videos']}"]
    if hands in ('both', 'left'):
        lines.append(
            f"left  max gap: {l['max_gap']}f  avg missing: {l['avg_missing']:.1f}f  "
            f"avg gaps/vid: {l['avg_gaps']:.2f}  affected: {l['pct_vids_affected']:.1f}%"
        )
    if hands in ('both', 'right'):
        lines.append(
            f"right max gap: {r['max_gap']}f  avg missing: {r['avg_missing']:.1f}f  "
            f"avg gaps/vid: {r['avg_gaps']:.2f}  affected: {r['pct_vids_affected']:.1f}%"
        )
    return '\n'.join(lines)



def _sort_records(records: List[Dict], sort_by: str, sort_desc: bool) -> List[Dict]:
    """sort records by name or frame length"""
    if sort_by == 'length':
        return sorted(records, key=lambda r: len(r['left_detected']), reverse=sort_desc)
    return sorted(records, key=lambda r: r['name'], reverse=sort_desc)

# ---------- rendering helpers ----------

def _bool_to_broken_barh(detected: List[bool]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Split a bool list into (detected_segments, missing_segments).
    Each segment is (start, width) for broken_barh.
    """
    detected_segs = []
    missing_segs  = []
    n = len(detected)
    i = 0
    while i < n:
        val = detected[i]
        start = i
        while i < n and detected[i] == val:
            i += 1
        seg = (start, i - start)
        if val:
            detected_segs.append(seg)
        else:
            missing_segs.append(seg)
    return detected_segs, missing_segs


def _draw_hand_bar(ax, y_bottom: float, detected: List[bool], color: str):
    """draw a single hand bar at y_bottom"""
    det_segs, mis_segs = _bool_to_broken_barh(detected)
    if mis_segs:
        ax.broken_barh(mis_segs, (y_bottom, BAR_HEIGHT), facecolors=COLOR_MISSING, linewidth=0)
    if det_segs:
        ax.broken_barh(det_segs, (y_bottom, BAR_HEIGHT), facecolors=color, linewidth=0)


# ---------- main plot function ----------

def plot_missing_hands(
    data_dir:      str,
    n_videos:      int           = 20,
    selection:     str           = 'random',  # 'random' or 'range'
    range_start:   int           = 0,
    range_end:     Optional[int] = None,
    output_path:   Optional[str] = None,
    figsize_scale: float         = 1.0,
    hands:         str           = 'both',    # 'both', 'left', 'right'
    bar_gap:       float         = BAR_GAP,    # gap between left/right bars of same video
    video_gap:     float         = VIDEO_GAP,  # gap between videos
    pct_mode:      str           = 'general',  # 'general' or 'intra'
    sort_by:       str           = 'name',     # 'name' or 'length'
    sort_desc:     bool          = False,
    csv_path:      Optional[str] = None,
):
    """
    Plot hand detection coverage for a subset of Phoenix JSON files.

    Args:
        data_dir:      directory containing JSON files
        n_videos:      how many videos to plot
        selection:     'random' or 'range'
        range_start:   start index when selection='range'
        range_end:     end index (inclusive) when selection='range'; None = range_start+n_videos
        output_path:   if set, save figure here instead of showing it
        figsize_scale: scale factor for figure size
        hands:         which hands to show - 'both', 'left', or 'right'
    """
    if hands not in ('both', 'left', 'right'):
        raise ValueError(f'hands must be "both", "left", or "right", got {hands!r}')

    data_dir = Path(data_dir)
    all_jsons = sorted(data_dir.glob('*.json'))

    if not all_jsons:
        raise FileNotFoundError(f'no JSON files in {data_dir}')

    if selection == 'random':
        chosen = random.sample(all_jsons, min(n_videos, len(all_jsons)))
        chosen = sorted(chosen)  # consistent y-order after random pick
    elif selection == 'range':
        end = range_end if range_end is not None else range_start + n_videos
        chosen = all_jsons[range_start:end + 1]
    else:
        raise ValueError(f'selection must be "random" or "range", got {selection!r}')

    records = []
    for jf in chosen:
        try:
            ld, rd = load_hand_detection(jf)
            records.append({'left_detected': ld, 'right_detected': rd, 'name': jf.stem})
        except Exception as e:
            print(f'skipping {jf.name}: {e}')

    if not records:
        raise RuntimeError('no valid JSON files could be loaded')

    records    = _sort_records(records, sort_by, sort_desc)
    stats      = compute_stats(records)
    max_frames = max(len(r['left_detected']) for r in records)

    show_left  = hands in ('both', 'left')
    show_right = hands in ('both', 'right')
    both       = show_left and show_right

    # row height depends on how many bars per video
    row_height = BAR_HEIGHT * (2 if both else 1) + bar_gap * (1 if both else 0) + video_gap

    n     = len(records)
    fig_h = max(4.0, n * row_height * figsize_scale + 1.8)
    fig_w = max(14.0, max_frames / 30 * figsize_scale)  # ~1 inch per 30 frames

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ytick_positions = []
    ytick_labels    = []

    pct_labels = []  # right-axis labels, one per video

    for idx, rec in enumerate(records):
        ld = rec['left_detected']
        rd = rec['right_detected']

        base = (n - 1 - idx) * row_height  # top video highest

        if both:
            y_left  = base + BAR_HEIGHT + bar_gap  # left on top
            y_right = base
            center  = base + BAR_HEIGHT + bar_gap / 2
        else:
            y_left = y_right = base
            center = base + BAR_HEIGHT / 2

        if show_left:
            _draw_hand_bar(ax, y_left,  ld, COLOR_LEFT_DETECTED)
        if show_right:
            _draw_hand_bar(ax, y_right, rd, COLOR_RIGHT_DETECTED)

        ytick_positions.append(center)
        ytick_labels.append(rec['name'])

        # compute detection % for secondary axis label
        if pct_mode == 'intra':
            s, e = _get_interior_range(ld, rd)
            l_range = ld[s:e + 1] if ld else []
            r_range = rd[s:e + 1] if rd else []
        else:
            l_range, r_range = ld, rd

        def _pct(seq):
            return sum(seq) / len(seq) * 100 if seq else 0.0

        if show_left and show_right:
            pct_labels.append(f'L:{_pct(l_range):.0f}%  R:{_pct(r_range):.0f}%')
        elif show_left:
            pct_labels.append(f'{_pct(l_range):.0f}%')
        else:
            pct_labels.append(f'{_pct(r_range):.0f}%')

    ax.set_xlim(0, max_frames)
    ax.set_ylim(-video_gap / 2, n * row_height)
    ax.set_xlabel('frame (timestep)', fontsize=10)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ytick_positions)
    ax2.set_yticklabels(pct_labels, fontsize=7)
    ax2.tick_params(axis='y', length=0)  # no tick marks, just labels
    ax2.set_ylabel(
        f'detection % ({"interior" if pct_mode == "intra" else "all"} frames)',
        fontsize=8
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(max(1, max_frames // 20)))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(max(1, max_frames // 100)))
    ax.grid(axis='x', which='major', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.grid(axis='x', which='minor', linestyle=':', linewidth=0.2, alpha=0.3)

    legend_handles = []
    if show_left:
        legend_handles.append(mpatches.Patch(color=COLOR_LEFT_DETECTED,  label='left hand detected'))
    if show_right:
        legend_handles.append(mpatches.Patch(color=COLOR_RIGHT_DETECTED, label='right hand detected'))
    legend_handles.append(mpatches.Patch(color=COLOR_MISSING, label='not detected'))
    ax.legend(handles=legend_handles, loc='lower right', fontsize=9, framealpha=0.9)

    title = _format_stats_title(stats, hands)
    fig.suptitle(title, fontsize=9, family='monospace', y=0.99, va='top',
                 ha='center', linespacing=1.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if csv_path:
        _export_missing_hands_csv(records, csv_path)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'saved to {output_path}')
    else:
        plt.show()

    plt.close(fig)



# ---------- cumulative plot ----------

def _draw_histogram_subplot(ax, det_count, mis_count, coverage, color, label, max_frames,
                            relative=False, min_coverage=1):
    """
    Stacked bars per frame: detected (dark) on bottom, missing (gray) on top.
    relative=True: bars scaled to 100%, y-axis is percentage.
    Percentage of detected annotated on top of every Nth bar (only in raw mode).
    """
    frames = np.arange(max_frames)

    with np.errstate(divide='ignore', invalid='ignore'):
        det_vals = np.where(coverage > 0, det_count / coverage * 100, 0.0)
        mis_vals = np.where(coverage > 0, mis_count / coverage * 100, 0.0)

    mask = coverage >= min_coverage  # frames with too few videos are hidden

    if relative:
        plot_det = np.where(mask, det_vals, 0.0)
        plot_mis = np.where(mask, mis_vals, 0.0)
    else:
        plot_det = np.where(mask, det_count.astype(float), 0.0)
        plot_mis = np.where(mask, mis_count.astype(float), 0.0)

    ax.bar(frames, plot_det, width=1, color=color,        align='edge', label='detected',     zorder=2)
    ax.bar(frames, plot_mis, width=1, color=COLOR_MISSING, align='edge', label='not detected', zorder=2,
           bottom=plot_det)

    # shade frames with insufficient coverage
    if not mask.all():
        for t in np.where(~mask)[0]:
            ax.axvspan(t, t + 1, color='white', alpha=0.6, zorder=3, linewidth=0)

    ax.set_ylabel('detection %' if relative else 'video count', fontsize=9)
    ax.set_xlim(0, max_frames)
    ax.set_title(label, fontsize=10, loc='left', pad=3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(max(1, max_frames // 20)))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(max(1, max_frames // 100)))
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.4, alpha=0.4, zorder=0)
    ax.grid(axis='x', which='major', linestyle='--', linewidth=0.4, alpha=0.3, zorder=0)

    if relative:
        ax.set_ylim(0, 115)  # headroom above 100%
        ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    else:
        # annotate % on top of full bar every N frames
        step  = max(1, max_frames // 40)
        total = plot_det + plot_mis
        for t in range(0, max_frames, step):
            if coverage[t] == 0:
                continue
            ax.text(
                t + 0.5, total[t] + ax.get_ylim()[1] * 0.01,
                f'{det_vals[t]:.0f}%',
                ha='center', va='bottom', fontsize=6, rotation=90, color=color,
            )

    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # coverage line on secondary axis so user sees how many vids contribute per frame
    ax2 = ax.twinx()
    ax2.plot(np.arange(max_frames) + 0.5, coverage, color='black', linewidth=0.8,
             alpha=0.35, linestyle=':', label='videos contributing')
    ax2.set_ylabel('videos at frame', fontsize=8, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
    ax2.set_ylim(0, coverage.max() * 2.5)  # keep line in lower portion


def plot_missing_hands_cumulative(
    data_dir:      str,
    n_videos:      int           = None,
    selection:     str           = 'random',
    range_start:   int           = 0,
    range_end:     Optional[int] = None,
    output_path:   Optional[str] = None,
    figsize_scale: float         = 1.0,
    hands:         str           = 'both',
    relative:      bool          = False,
    min_coverage:  int           = 1,    # frames with fewer videos are hidden
    sort_by:       str           = 'name',
    sort_desc:     bool          = False,
):
    """
    Histogram of hand detection counts per frame across all analyzed videos.
    Each frame is a bucket: grouped bars for detected (dark) and missing (gray).
    Percentage of detected annotated above detected bars.
    """
    if hands not in ('both', 'left', 'right'):
        raise ValueError(f'hands must be "both", "left", or "right", got {hands!r}')

    data_dir  = Path(data_dir)
    all_jsons = sorted(data_dir.glob('*.json'))

    if not all_jsons:
        raise FileNotFoundError(f'no JSON files in {data_dir}')

    if n_videos is None:
        chosen = all_jsons
    elif selection == 'random':
        chosen = sorted(random.sample(all_jsons, min(n_videos, len(all_jsons))))
    elif selection == 'range':
        end    = range_end if range_end is not None else range_start + n_videos
        chosen = all_jsons[range_start:end + 1]
    else:
        raise ValueError(f'selection must be "random" or "range", got {selection!r}')

    records = []
    for jf in chosen:
        try:
            ld, rd = load_hand_detection(jf)
            records.append({'left_detected': ld, 'right_detected': rd})
        except Exception as e:
            print(f'skipping {jf.name}: {e}')

    if not records:
        raise RuntimeError('no valid JSON files could be loaded')

    records    = _sort_records(records, sort_by, sort_desc)
    n_vids     = len(records)
    max_frames = max(len(r['left_detected']) for r in records)

    left_det_count  = np.zeros(max_frames, dtype=np.int64)
    left_mis_count  = np.zeros(max_frames, dtype=np.int64)
    right_det_count = np.zeros(max_frames, dtype=np.int64)
    right_mis_count = np.zeros(max_frames, dtype=np.int64)
    coverage        = np.zeros(max_frames, dtype=np.int64)

    for r in records:
        n = len(r['left_detected'])
        coverage[:n] += 1
        for t, (ld, rd) in enumerate(zip(r['left_detected'], r['right_detected'])):
            left_det_count[t]  += int(ld)
            left_mis_count[t]  += int(not ld)
            right_det_count[t] += int(rd)
            right_mis_count[t] += int(not rd)

    show_left  = hands in ('both', 'left')
    show_right = hands in ('both', 'right')
    n_subplots = 2 if (show_left and show_right) else 1

    fig_w = max(14.0, max_frames / 15 * figsize_scale)
    fig_h = max(4.0, 3.5 * n_subplots * figsize_scale)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(fig_w, fig_h), sharex=True)
    if n_subplots == 1:
        axes = [axes]

    ax_idx = 0
    if show_left:
        _draw_histogram_subplot(
            axes[ax_idx], left_det_count, left_mis_count,
            coverage, COLOR_LEFT_DETECTED, 'left hand', max_frames,
            relative=relative, min_coverage=min_coverage
        )
        ax_idx += 1
    if show_right:
        _draw_histogram_subplot(
            axes[ax_idx], right_det_count, right_mis_count,
            coverage, COLOR_RIGHT_DETECTED, 'right hand', max_frames,
            relative=relative, min_coverage=min_coverage
        )

    axes[-1].set_xlabel('frame (timestep)', fontsize=10)

    def _overall_pct(det, mis):
        total = int(det.sum()) + int(mis.sum())
        return int(det.sum()) / total * 100 if total > 0 else 0.0

    lines = [f'videos: {n_vids}']
    if show_left:
        lines.append(
            f'left   detected: {int(left_det_count.sum())}  '
            f'missing: {int(left_mis_count.sum())}  '
            f'overall detection rate: {_overall_pct(left_det_count, left_mis_count):.1f}%'
        )
    if show_right:
        lines.append(
            f'right  detected: {int(right_det_count.sum())}  '
            f'missing: {int(right_mis_count.sum())}  '
            f'overall detection rate: {_overall_pct(right_det_count, right_mis_count):.1f}%'
        )
    fig.suptitle('\n'.join(lines), fontsize=9, family='monospace', y=0.99, va='top',
                 ha='center', linespacing=1.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'saved to {output_path}')
    else:
        plt.show()

    plt.close(fig)


# ---------- gap distribution plot ----------

def plot_gap_distribution(
    data_dir:      str,
    n_videos:      int           = None,
    selection:     str           = 'random',
    range_start:   int           = 0,
    range_end:     Optional[int] = None,
    output_path:   Optional[str] = None,
    figsize_scale: float         = 1.0,
    hands:         str           = 'both',
    sort_by:       str           = 'name',
    sort_desc:     bool          = False,
    csv_path:      Optional[str] = None,
):
    """
    Histogram of consecutive missing frame run lengths across all analyzed videos.
    X = gap length (number of consecutive missing frames), Y = how many times it occurred.
    Only interior gaps are counted (boundary silence excluded).
    """
    if hands not in ('both', 'left', 'right'):
        raise ValueError(f'hands must be "both", "left", or "right", got {hands!r}')

    data_dir  = Path(data_dir)
    all_jsons = sorted(data_dir.glob('*.json'))

    if not all_jsons:
        raise FileNotFoundError(f'no JSON files in {data_dir}')

    if n_videos is None:
        chosen = all_jsons
    elif selection == 'random':
        chosen = sorted(random.sample(all_jsons, min(n_videos, len(all_jsons))))
    elif selection == 'range':
        end    = range_end if range_end is not None else range_start + n_videos
        chosen = all_jsons[range_start:end + 1]
    else:
        raise ValueError(f'selection must be "random" or "range", got {selection!r}')

    records = []
    for jf in chosen:
        try:
            ld, rd = load_hand_detection(jf)
            records.append({'left_detected': ld, 'right_detected': rd, 'name': jf.stem})
        except Exception as e:
            print(f'skipping {jf.name}: {e}')

    if not records:
        raise RuntimeError('no valid JSON files could be loaded')

    records = _sort_records(records, sort_by, sort_desc)
    n_vids  = len(records)

    # collect all interior gap lengths per hand
    from collections import Counter
    left_gap_counts  = Counter()
    right_gap_counts = Counter()

    for r in records:
        ld, rd = r['left_detected'], r['right_detected']
        s, e   = _get_interior_range(ld, rd)
        for gap in _find_gaps(ld, s, e):
            left_gap_counts[gap] += 1
        for gap in _find_gaps(rd, s, e):
            right_gap_counts[gap] += 1

    show_left  = hands in ('both', 'left')
    show_right = hands in ('both', 'right')
    n_subplots = 2 if (show_left and show_right) else 1

    # x range: cover all observed gap lengths
    all_lengths = set()
    if show_left:
        all_lengths.update(left_gap_counts.keys())
    if show_right:
        all_lengths.update(right_gap_counts.keys())

    if not all_lengths:
        print('no interior gaps found in the selected videos')
        return

    max_gap = max(all_lengths)
    xs      = np.arange(1, max_gap + 1)

    fig_w = max(10.0, max_gap / 3 * figsize_scale)
    fig_h = max(4.0, 3.5 * n_subplots * figsize_scale)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(fig_w, fig_h), sharex=True)
    if n_subplots == 1:
        axes = [axes]

    def _draw_gap_ax(ax, gap_counts, color, label):
        ys = np.array([gap_counts.get(x, 0) for x in xs])
        ax.bar(xs, ys, width=0.8, color=color, align='center', zorder=2)

        # value labels on top of each bar
        for x, y in zip(xs, ys):
            if y > 0:
                ax.text(x, y + ax.get_ylim()[1] * 0.01, str(y),
                        ha='center', va='bottom', fontsize=7, color=color)

        ax.set_title(label, fontsize=10, loc='left', pad=3)
        ax.set_ylabel('occurrences', fontsize=9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(max(1, max_gap // 20)))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(axis='y', which='major', linestyle='--', linewidth=0.4, alpha=0.4, zorder=0)
        ax.set_xlim(0.5, max_gap + 0.5)

        total_gaps   = sum(gap_counts.values())
        total_frames = sum(k * v for k, v in gap_counts.items())
        ax.set_xlabel('')  # will be set on bottom axis only
        return total_gaps, total_frames

    ax_idx = 0
    lines  = [f'videos: {n_vids}  |  interior gaps only']

    if show_left:
        # bar labels need ylim set first - draw, then annotate
        ys = np.array([left_gap_counts.get(x, 0) for x in xs])
        axes[ax_idx].bar(xs, ys, width=0.8, color=COLOR_LEFT_DETECTED, align='center', zorder=2)
        axes[ax_idx].set_title('left hand', fontsize=10, loc='left', pad=3)
        axes[ax_idx].set_ylabel('occurrences', fontsize=9)
        axes[ax_idx].xaxis.set_major_locator(ticker.MultipleLocator(max(1, max_gap // 20)))
        axes[ax_idx].xaxis.set_minor_locator(ticker.MultipleLocator(1))
        axes[ax_idx].grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4, zorder=0)
        axes[ax_idx].set_xlim(0.5, max_gap + 0.5)
        ymax = axes[ax_idx].get_ylim()[1]
        for x, y in zip(xs, ys):
            if y > 0:
                axes[ax_idx].text(x, y + ymax * 0.01, str(y),
                                  ha='center', va='bottom', fontsize=7,
                                  color=COLOR_LEFT_DETECTED)
        total_l = sum(left_gap_counts.values())
        frames_l = sum(k * v for k, v in left_gap_counts.items())
        lines.append(
            f'left   total gaps: {total_l}  total missing frames in gaps: {frames_l}  ' +
            f'max gap: {max(left_gap_counts) if left_gap_counts else 0}  ' +
            f'most common gap: {left_gap_counts.most_common(1)[0][0] if left_gap_counts else "-"}f ' +
            f'({left_gap_counts.most_common(1)[0][1] if left_gap_counts else 0}x)'
        )
        ax_idx += 1

    if show_right:
        ys = np.array([right_gap_counts.get(x, 0) for x in xs])
        axes[ax_idx].bar(xs, ys, width=0.8, color=COLOR_RIGHT_DETECTED, align='center', zorder=2)
        axes[ax_idx].set_title('right hand', fontsize=10, loc='left', pad=3)
        axes[ax_idx].set_ylabel('occurrences', fontsize=9)
        axes[ax_idx].xaxis.set_major_locator(ticker.MultipleLocator(max(1, max_gap // 20)))
        axes[ax_idx].xaxis.set_minor_locator(ticker.MultipleLocator(1))
        axes[ax_idx].grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4, zorder=0)
        axes[ax_idx].set_xlim(0.5, max_gap + 0.5)
        ymax = axes[ax_idx].get_ylim()[1]
        for x, y in zip(xs, ys):
            if y > 0:
                axes[ax_idx].text(x, y + ymax * 0.01, str(y),
                                  ha='center', va='bottom', fontsize=7,
                                  color=COLOR_RIGHT_DETECTED)
        total_r = sum(right_gap_counts.values())
        frames_r = sum(k * v for k, v in right_gap_counts.items())
        lines.append(
            f'right  total gaps: {total_r}  total missing frames in gaps: {frames_r}  ' +
            f'max gap: {max(right_gap_counts) if right_gap_counts else 0}  ' +
            f'most common gap: {right_gap_counts.most_common(1)[0][0] if right_gap_counts else "-"}f ' +
            f'({right_gap_counts.most_common(1)[0][1] if right_gap_counts else 0}x)'
        )

    axes[-1].set_xlabel('consecutive missing frames (gap length)', fontsize=10)

    fig.suptitle('\n'.join(lines), fontsize=9, family='monospace', y=0.99, va='top',
                 ha='center', linespacing=1.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if csv_path:
        _export_gap_distribution_csv(left_gap_counts, right_gap_counts, csv_path)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'saved to {output_path}')
    else:
        plt.show()

    plt.close(fig)


# ---------- csv export ----------

def _export_missing_hands_csv(records: List[Dict], csv_path: str):
    """
    One row per video: name, total frames, per-hand detected/missing counts
    and detection % for both general (all frames) and intra (interior only).
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'video', 'total_frames',
            'left_detected', 'left_missing', 'left_pct_general', 'left_pct_intra',
            'right_detected', 'right_missing', 'right_pct_general', 'right_pct_intra',
            'interior_start', 'interior_end',
        ])
        for r in records:
            ld, rd = r['left_detected'], r['right_detected']
            n      = len(ld)
            s, e   = _get_interior_range(ld, rd)

            def _pct(seq):
                return round(sum(seq) / len(seq) * 100, 2) if seq else 0.0

            intra_l = ld[s:e + 1]
            intra_r = rd[s:e + 1]

            writer.writerow([
                r['name'], n,
                sum(ld), n - sum(ld), _pct(ld), _pct(intra_l),
                sum(rd), n - sum(rd), _pct(rd), _pct(intra_r),
                s, e,
            ])
    print(f'saved csv to {csv_path}')


def _export_gap_distribution_csv(left_gap_counts, right_gap_counts, csv_path: str):
    """One row per gap length: gap_length, left_count, right_count."""
    all_lengths = sorted(set(left_gap_counts) | set(right_gap_counts))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['gap_length', 'left_count', 'right_count'])
        for length in all_lengths:
            writer.writerow([length, left_gap_counts.get(length, 0), right_gap_counts.get(length, 0)])
    print(f'saved csv to {csv_path}')


# ---------- additional statistics ----------

def _load_records(data_dir: str, n_videos, selection, range_start, range_end, sort_by, sort_desc, seed=None):
    """shared record loading used by the new stat subcommands"""
    if seed is not None:
        random.seed(seed)
    data_dir  = Path(data_dir)
    all_jsons = sorted(data_dir.glob('*.json'))
    if not all_jsons:
        raise FileNotFoundError(f'no JSON files in {data_dir}')
    if n_videos is None:
        chosen = all_jsons
    elif selection == 'random':
        chosen = sorted(random.sample(all_jsons, min(n_videos, len(all_jsons))))
    elif selection == 'range':
        end    = range_end if range_end is not None else range_start + n_videos
        chosen = all_jsons[range_start:end + 1]
    else:
        raise ValueError(f'bad selection {selection!r}')
    records = []
    for jf in chosen:
        try:
            ld, rd = load_hand_detection(jf)
            records.append({'left_detected': ld, 'right_detected': rd, 'name': jf.stem})
        except Exception as e:
            print(f'skipping {jf.name}: {e}')
    if not records:
        raise RuntimeError('no valid JSON files could be loaded')
    return _sort_records(records, sort_by, sort_desc)


def _common_args_dict(args):
    return dict(
        data_dir    = args.data_dir,
        n_videos    = args.n_videos,
        selection   = args.selection,
        range_start = args.range_start,
        range_end   = args.range_end,
        output_path = args.output,
        figsize_scale = args.figsize_scale,
        sort_by     = args.sort_by,
        sort_desc   = args.sort_desc,
    )


def _add_stat_args(p):
    """shared args for the 4 stat subcommands"""
    _add_common_args(p)
    p.add_argument('--n', type=int, default=None, dest='n_videos',
                   help='number of videos to include (default: all)')
    p.add_argument('--selection', choices=['random', 'range'], default='random',
                   help='how to pick videos when --n is set (default: random)')
    p.add_argument('--range_start', type=int, default=0)
    p.add_argument('--range_end',   type=int, default=None)
    p.add_argument('--scale', type=float, default=1.0, dest='figsize_scale')
    p.add_argument('--sort-by', choices=['name', 'length'], default='name', dest='sort_by')
    p.add_argument('--sort-desc', action='store_true', default=False, dest='sort_desc')


# ---------- CLI ----------

def _add_common_args(p: argparse.ArgumentParser):
    """args shared across subcommands that work with a data directory"""
    p.add_argument('data_dir', help='directory containing JSON landmark files')
    p.add_argument('--output', default=None,
                   help='save figure to this path instead of displaying it')
    p.add_argument('--seed', type=int, default=None,
                   help='random seed for reproducible random selection')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='data quality checks for Phoenix sign language JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest='command', metavar='COMMAND')
    sub.required = True

    # -- plot-missing-hands --
    pmh = sub.add_parser(
        'plot-missing-hands',
        help='plot per-frame hand detection coverage as horizontal bars',
    )
    _add_common_args(pmh)
    pmh.add_argument('--n', type=int, default=20, dest='n_videos',
                     help='number of videos to display (default: 20)')
    pmh.add_argument('--selection', choices=['random', 'range'], default='random',
                     help='how to pick videos (default: random)')
    pmh.add_argument('--range_start', type=int, default=0,
                     help='start index for range selection')
    pmh.add_argument('--range_end', type=int, default=None,
                     help='end index (inclusive) for range selection')
    pmh.add_argument('--scale', type=float, default=1.0, dest='figsize_scale',
                     help='scale factor for figure dimensions (default: 1.0)')
    pmh.add_argument('--bar-gap', type=float, default=BAR_GAP, dest='bar_gap',
                     help=f'gap between left/right bars of same video (default: {BAR_GAP})')
    pmh.add_argument('--video-gap', type=float, default=VIDEO_GAP, dest='video_gap',
                     help=f'vertical spacing between videos (default: {VIDEO_GAP})')
    pmh.add_argument('--hands', choices=['both', 'left', 'right'], default='both',
                     help='which hands to show on the graph (default: both)')
    pmh.add_argument('--pct-mode', choices=['general', 'intra'], default='general',
                     dest='pct_mode',
                     help='percentage on right axis: all frames (general) or interior only (intra)')
    pmh.add_argument('--sort-by', choices=['name', 'length'], default='name', dest='sort_by',
                     help='sort videos by name or frame length (default: name)')
    pmh.add_argument('--sort-desc', action='store_true', default=False, dest='sort_desc',
                     help='reverse sort order')
    pmh.add_argument('--csv', default=None, dest='csv_path',
                     help='export data to this csv path')

    # -- plot-missing-hands-cumulative --
    pmhc = sub.add_parser(
        'plot-missing-hands-cumulative',
        help='cumulative detected/missing frame counts across all videos',
    )
    _add_common_args(pmhc)
    pmhc.add_argument('--n', type=int, default=None, dest='n_videos',
                      help='number of videos to include (default: all)')
    pmhc.add_argument('--selection', choices=['random', 'range'], default='random',
                      help='how to pick videos when --n is set (default: random)')
    pmhc.add_argument('--range_start', type=int, default=0,
                      help='start index for range selection')
    pmhc.add_argument('--range_end', type=int, default=None,
                      help='end index (inclusive) for range selection')
    pmhc.add_argument('--scale', type=float, default=1.0, dest='figsize_scale',
                      help='scale factor for figure dimensions (default: 1.0)')
    pmhc.add_argument('--hands', choices=['both', 'left', 'right'], default='both',
                      help='which hands to show on the graph (default: both)')
    pmhc.add_argument('--relative', action='store_true', default=False,
                      help='show bars as percentage of videos at that frame, y-axis shows detection percentage')
    pmhc.add_argument('--min-coverage', type=int, default=1, dest='min_coverage',
                      help='hide frames where fewer than N videos contribute (default: 1)')
    pmhc.add_argument('--sort-by', choices=['name', 'length'], default='name', dest='sort_by',
                     help='sort videos by name or frame length (default: name)')
    pmhc.add_argument('--sort-desc', action='store_true', default=False, dest='sort_desc',
                     help='reverse sort order')

    # -- plot-gap-distribution --
    pgd = sub.add_parser(
        'plot-gap-distribution',
        help='histogram of consecutive missing frame run lengths across all analyzed videos',
    )
    _add_common_args(pgd)
    pgd.add_argument('--n', type=int, default=None, dest='n_videos',
                     help='number of videos to include (default: all)')
    pgd.add_argument('--selection', choices=['random', 'range'], default='random',
                     help='how to pick videos when --n is set (default: random)')
    pgd.add_argument('--range_start', type=int, default=0,
                     help='start index for range selection')
    pgd.add_argument('--range_end', type=int, default=None,
                     help='end index (inclusive) for range selection')
    pgd.add_argument('--scale', type=float, default=1.0, dest='figsize_scale',
                     help='scale factor for figure dimensions (default: 1.0)')
    pgd.add_argument('--hands', choices=['both', 'left', 'right'], default='both',
                     help='which hands to show (default: both)')
    pgd.add_argument('--sort-by', choices=['name', 'length'], default='name', dest='sort_by',
                     help='sort videos by name or frame length (default: name)')
    pgd.add_argument('--sort-desc', action='store_true', default=False, dest='sort_desc',
                     help='reverse sort order')
    pgd.add_argument('--csv', default=None, dest='csv_path',
                     help='export data to this csv path')
    return p



def _run_plot_missing_hands(args):
    if args.seed is not None:
        random.seed(args.seed)
    plot_missing_hands(
        data_dir      = args.data_dir,
        n_videos      = args.n_videos,
        selection     = args.selection,
        range_start   = args.range_start,
        range_end     = args.range_end,
        output_path   = args.output,
        figsize_scale = args.figsize_scale,
        hands         = args.hands,
        pct_mode      = args.pct_mode,
        sort_by       = args.sort_by,
        sort_desc     = args.sort_desc,
        csv_path      = args.csv_path,
        bar_gap       = args.bar_gap,
        video_gap     = args.video_gap,
    )


def _run_plot_missing_hands_cumulative(args):
    if args.seed is not None:
        random.seed(args.seed)
    plot_missing_hands_cumulative(
        data_dir      = args.data_dir,
        n_videos      = args.n_videos,
        selection     = args.selection,
        range_start   = args.range_start,
        range_end     = args.range_end,
        output_path   = args.output,
        figsize_scale = args.figsize_scale,
        hands         = args.hands,
        relative       = args.relative,
        min_coverage   = args.min_coverage,
        sort_by        = args.sort_by,
        sort_desc      = args.sort_desc,
    )



def _run_plot_gap_distribution(args):
    if args.seed is not None:
        random.seed(args.seed)
    plot_gap_distribution(
        data_dir      = args.data_dir,
        n_videos      = args.n_videos,
        selection     = args.selection,
        range_start   = args.range_start,
        range_end     = args.range_end,
        output_path   = args.output,
        figsize_scale = args.figsize_scale,
        hands         = args.hands,
        sort_by       = args.sort_by,
        sort_desc     = args.sort_desc,
    )


_COMMAND_HANDLERS = {
    'plot-missing-hands':            _run_plot_missing_hands,
    'plot-missing-hands-cumulative': _run_plot_missing_hands_cumulative,
    'plot-gap-distribution':         _run_plot_gap_distribution,
}


if __name__ == '__main__':
    args = _build_parser().parse_args()
    _COMMAND_HANDLERS[args.command](args)