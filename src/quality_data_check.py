#!/usr/bin/env python3
"""
Data quality visualization for sign language JSON landmark files.
Currently supports Phoenix-format JSONs only.
"""

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
    left_gap_lengths  = []
    right_gap_lengths = []
    left_gaps_per_vid  = []
    right_gaps_per_vid = []
    left_missing_interior  = []
    right_missing_interior = []
    vids_with_left_gap  = 0
    vids_with_right_gap = 0

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
            'max_gap':       _safe(max, left_gap_lengths),
            'avg_missing':   _safe(np.mean, left_missing_interior),
            'avg_gaps':      _safe(np.mean, left_gaps_per_vid),
            'pct_vids_affected': vids_with_left_gap / n * 100 if n else 0,
        },
        'right': {
            'max_gap':       _safe(max, right_gap_lengths),
            'avg_missing':   _safe(np.mean, right_missing_interior),
            'avg_gaps':      _safe(np.mean, right_gaps_per_vid),
            'pct_vids_affected': vids_with_right_gap / n * 100 if n else 0,
        },
    }


def _format_stats_title(stats: Dict) -> str:
    l = stats['left']
    r = stats['right']
    lines = [
        f"videos: {stats['n_videos']}  |  "
        f"left  max gap: {l['max_gap']}f  avg missing: {l['avg_missing']:.1f}f  "
        f"avg gaps/vid: {l['avg_gaps']:.2f}  affected: {l['pct_vids_affected']:.1f}%",

        f"right max gap: {r['max_gap']}f  avg missing: {r['avg_missing']:.1f}f  "
        f"avg gaps/vid: {r['avg_gaps']:.2f}  affected: {r['pct_vids_affected']:.1f}%",
    ]
    return '\n'.join(lines)


# ---------- rendering helpers ----------

def _bool_to_broken_barh(detected: List[bool]) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    """
    Split a bool list into (detected_segments, missing_segments)
    each segment is (start, width) for broken_barh
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
    data_dir:    str,
    n_videos:    int          = 20,
    selection:   str          = 'random',  # 'random' or 'range'
    range_start: int          = 0,
    range_end:   Optional[int] = None,
    output_path: Optional[str] = None,
    figsize_scale: float      = 1.0,
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
    """
    data_dir = Path(data_dir)
    all_jsons = sorted(data_dir.glob('*.json'))

    if not all_jsons:
        raise FileNotFoundError(f'no JSON files in {data_dir}')

    # pick subset
    if selection == 'random':
        chosen = random.sample(all_jsons, min(n_videos, len(all_jsons)))
        chosen = sorted(chosen)  # consistent y-order after random pick
    elif selection == 'range':
        end = range_end if range_end is not None else range_start + n_videos
        chosen = all_jsons[range_start:end + 1]
    else:
        raise ValueError(f'selection must be "random" or "range", got {selection!r}')

    # load data
    records = []
    labels  = []
    for jf in chosen:
        try:
            ld, rd = load_hand_detection(jf)
            records.append({'left_detected': ld, 'right_detected': rd, 'name': jf.stem})
            labels.append(jf.stem)
        except Exception as e:
            print(f'skipping {jf.name}: {e}')

    if not records:
        raise RuntimeError('no valid JSON files could be loaded')

    stats = compute_stats(records)
    max_frames = max(len(r['left_detected']) for r in records)

    n = len(records)
    row_height = BAR_HEIGHT * 2 + BAR_GAP + VIDEO_GAP
    fig_h = max(4.0, n * row_height * figsize_scale + 1.8)
    fig_w = max(14.0, max_frames / 30 * figsize_scale)  # ~1 inch per 30 frames

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ytick_positions = []
    ytick_labels    = []

    for idx, rec in enumerate(records):
        ld = rec['left_detected']
        rd = rec['right_detected']

        # y positions: top video is highest, so invert
        base = (n - 1 - idx) * row_height

        y_left  = base + BAR_HEIGHT + BAR_GAP  # left hand bar on top
        y_right = base                          # right hand bar below

        _draw_hand_bar(ax, y_left,  ld, COLOR_LEFT_DETECTED)
        _draw_hand_bar(ax, y_right, rd, COLOR_RIGHT_DETECTED)

        # tick at vertical center of the two bars
        center = base + BAR_HEIGHT + BAR_GAP / 2
        ytick_positions.append(center)
        ytick_labels.append(rec['name'])

    # axes formatting
    ax.set_xlim(0, max_frames)
    ax.set_ylim(-VIDEO_GAP / 2, n * row_height)
    ax.set_xlabel('frame (timestep)', fontsize=10)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(max(1, max_frames // 20)))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(max(1, max_frames // 100)))
    ax.grid(axis='x', which='major', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.grid(axis='x', which='minor', linestyle=':', linewidth=0.2, alpha=0.3)

    # legend
    legend_handles = [
        mpatches.Patch(color=COLOR_LEFT_DETECTED,  label='left hand detected'),
        mpatches.Patch(color=COLOR_RIGHT_DETECTED, label='right hand detected'),
        mpatches.Patch(color=COLOR_MISSING,        label='not detected'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=9, framealpha=0.9)

    # stats header
    title = _format_stats_title(stats)
    fig.suptitle(title, fontsize=9, family='monospace', y=0.99, va='top',
                 ha='center', linespacing=1.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'saved to {output_path}')
    else:
        plt.show()

    plt.close(fig)


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
    )


_COMMAND_HANDLERS = {
    'plot-missing-hands': _run_plot_missing_hands,
}


if __name__ == '__main__':
    args = _build_parser().parse_args()
    _COMMAND_HANDLERS[args.command](args)