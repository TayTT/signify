#!/usr/bin/env python3
"""
Build an augmented training dataset from Phoenix-format JSON landmark files.

For every source file three augmented copies are always produced:
    1. flip
    2. jitter
    3. flip + jitter

For source files that contain any gloss appearing in fewer than
--rare-threshold distinct videos, one additional copy is produced:
    4. rotate + scale + jitter  (different seed for variety)

The output directory contains:
    - hard-links (or copies) of all original JSON files
    - all augmented JSON files
    - a combined annotations CSV covering every file in the directory

Usage:
    python src/build_augmented_dataset.py
        --json_dir   ./data/phoenix_jsons/phoenix_train
        --csv_path   ./data/annotations_phoenix/train_corpus.csv
        --output_dir ./data/phoenix_jsons/phoenix_train_aug
        --output_csv ./data/annotations_phoenix/train_corpus_aug.csv
        --rare-threshold 5
        --seed 42
"""

import json
import shutil
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quality_gloss_check import load_annotations, get_gloss_video_counts
from skeleton_augmentation import AugmentationConfig, process_file


# ---------- augmentation plan ----------

# baseline configs applied to every file
BASELINE_CONFIGS = [
    AugmentationConfig(flip=True),
    AugmentationConfig(jitter=True,  jitter_sigma=0.0075),
    AugmentationConfig(flip=True,    jitter=True, jitter_sigma=0.0075),
]

# extra configs applied to videos that contain rare glosses
RARE_CONFIG = AugmentationConfig(
    jitter=True, jitter_sigma=0.005,
    scale=True,  scale_min=0.85, scale_max=1.15,
    rotate=True, rotate_max=10.0,
)
RARE_CONFIG_FLIP = AugmentationConfig(
    jitter=True, jitter_sigma=0.005,
    scale=True,  scale_min=0.85, scale_max=1.15,
    rotate=True, rotate_max=10.0,
    flip=True,
)

RARE_SEED_OFFSET = 1000  # different seed from baseline to ensure distinct transformations


# ---------- helpers ----------

def _find_json(json_dir: Path, sample_id: str) -> Path | None:
    """exact stem match, then substring fallback (mirrors train_lstm.py logic)"""
    exact = json_dir / f'{sample_id}.json'
    if exact.exists():
        return exact
    for jf in json_dir.glob('*.json'):
        if sample_id in jf.stem or jf.stem in sample_id:
            return jf
    return None


def _copy_original(src: Path, dst_dir: Path) -> Path:
    """copy original json to output dir; skip if already there"""
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def _ids_with_rare_glosses(df, rare_glosses: Set[str]) -> Set[str]:
    """return set of sample ids whose annotation contains any rare gloss"""
    rare_ids = set()
    for _, row in df.iterrows():
        glosses = set(row['annotation'].split())
        if glosses & rare_glosses:
            rare_ids.add(row['id'])
    return rare_ids


# ---------- main ----------

def build_dataset(
    json_dir:        Path,
    csv_path:        Path,
    output_dir:      Path,
    output_csv:      Path,
    rare_threshold:  int  = 5,
    seed:            int  = 42,
    verbose:         bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_annotations(str(csv_path))
    video_counts = get_gloss_video_counts(df)

    rare_glosses = {g for g, c in video_counts.items() if c < rare_threshold}
    rare_ids     = _ids_with_rare_glosses(df, rare_glosses)

    if verbose:
        print(f'[WARNING!] New files will be added to the directory, without being it cleaned first! The csv file will be overriden and will not contain old data!')
        print(f'source videos  : {len(df)}')
        print(f'rare glosses   : {len(rare_glosses)} (fewer than {rare_threshold} videos)')
        print(f'videos selected for extra augmentation: {len(rare_ids)}')
        print(f'output dir     : {output_dir}')
        print()

    # rows for the output CSV: [id, folder, signer, annotation]
    csv_rows: List[Dict] = []

    n_orig = 0
    n_aug  = 0
    errors = 0

    for i, row in df.iterrows():
        sample_id  = row['id']
        annotation = row['annotation']
        folder     = row['folder']
        signer     = row['signer']

        src_json = _find_json(json_dir, sample_id)
        if src_json is None:
            if verbose:
                print(f'  [WARN] no JSON for {sample_id}, skipping')
            errors += 1
            continue

        # copy original
        _copy_original(src_json, output_dir)
        csv_rows.append({'id': sample_id, 'folder': folder,
                         'signer': signer, 'annotation': annotation})
        n_orig += 1

        # baseline augmentations
        for cfg_idx, cfg in enumerate(BASELINE_CONFIGS):
            file_seed = seed + i * 10 + cfg_idx
            try:
                out = process_file(src_json, output_dir, cfg, seed=file_seed)
                csv_rows.append({
                    'id':         out.stem,
                    'folder':     folder,
                    'signer':     signer,
                    'annotation': annotation,
                })
                n_aug += 1
            except Exception as e:
                if verbose:
                    print(f'  [ERROR] {src_json.name} cfg={cfg.tag()}: {e}')
                errors += 1

        # extra augmentation for videos with rare glosses
        if sample_id in rare_ids:
            for rare_idx, (rare_cfg, rare_seed) in enumerate([
                (RARE_CONFIG,                                                              seed + RARE_SEED_OFFSET + i),
                (RARE_CONFIG_FLIP,                                                            seed + RARE_SEED_OFFSET + i + 1),
            ]):
                try:
                    out = process_file(src_json, output_dir, rare_cfg, seed=rare_seed)
                    csv_rows.append({
                        'id':         out.stem,
                        'folder':     folder,
                        'signer':     signer,
                        'annotation': annotation,
                    })
                    n_aug += 1
                except Exception as e:
                    if verbose:
                        print(f'  [ERROR] {src_json.name} rare cfg={rare_cfg.tag()}: {e}')
                    errors += 1

        if verbose and (i + 1) % 100 == 0:
            print(f'  processed {i + 1}/{len(df)} videos...')

    # write combined CSV (pipe-separated, Phoenix format)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        f.write('id|folder|signer|annotation\n')
        for row in csv_rows:
            f.write(f"{row['id']}|{row['folder']}|{row['signer']}|{row['annotation']}\n")

    print()
    print(f'done.')
    print(f'  original files : {n_orig}')
    print(f'  augmented files: {n_aug}')
    print(f'  total entries  : {len(csv_rows)}')
    print(f'  errors/skipped : {errors}')
    print(f'  csv written to : {output_csv}')


# ---------- CLI ----------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='build augmented Phoenix training dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--json_dir',   required=True,
                   help='directory containing source JSON landmark files')
    p.add_argument('--csv_path',   required=True,
                   help='source annotations CSV (Phoenix format)')
    p.add_argument('--output_dir', required=True,
                   help='output directory for all JSON files (original + augmented)')
    p.add_argument('--output_csv', required=True,
                   help='output annotations CSV covering all files in output_dir')
    p.add_argument('--rare-threshold', type=int, default=5, dest='rare_threshold',
                   help='videos containing any gloss below this video-count get extra augmentation (default: 5)')
    p.add_argument('--seed', type=int, default=42,
                   help='base random seed (default: 42)')
    p.add_argument('--quiet', action='store_true',
                   help='suppress per-file progress output')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    build_dataset(
        json_dir       = Path(args.json_dir),
        csv_path       = Path(args.csv_path),
        output_dir     = Path(args.output_dir),
        output_csv     = Path(args.output_csv),
        rare_threshold = args.rare_threshold,
        seed           = args.seed,
        verbose        = not args.quiet,
    )