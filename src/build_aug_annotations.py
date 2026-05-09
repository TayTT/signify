#!/usr/bin/env python3
import re
import argparse
from pathlib import Path

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quality_gloss_check import load_annotations


# matches the _augNN_ separator produced by augment_to_npz.py
_AUG_PATTERN = re.compile(r'_aug\d{2}_.*$')


def _recover_original_stem(stem: str) -> str | None:
    m = _AUG_PATTERN.search(stem)
    if m is None:
        return None
    return stem[:m.start()]


def build_augmented_csv(
    aug_dir:    Path,
    source_csv: Path,
    output_csv: Path,
    verbose:    bool = True,
):
    source_df  = load_annotations(str(source_csv))
    lookup     = {row['id']: row for _, row in source_df.iterrows()}  # stem -> row

    npz_files  = sorted(aug_dir.glob('*.npz'))
    if not npz_files:
        raise FileNotFoundError(f'no NPZ files found in {aug_dir}')

    rows      = []
    skipped   = 0

    for npz in npz_files:
        orig_stem = _recover_original_stem(npz.stem)
        if orig_stem is None:
            if verbose:
                print(f'  [SKIP] no _augNN_ pattern in {npz.name}')
            skipped += 1
            continue

        src = lookup.get(orig_stem)
        if src is None:
            if verbose:
                print(f'  [SKIP] no source row for stem "{orig_stem}" ({npz.name})')
            skipped += 1
            continue

        rows.append({
            'id':         npz.stem,
            'folder':     src['folder'],
            'signer':     src['signer'],
            'annotation': src['annotation'],
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        f.write('id|folder|signer|annotation\n')
        for _, src_row in source_df.iterrows():  # originals first
            f.write(f"{src_row['id']}|{src_row['folder']}|{src_row['signer']}|{src_row['annotation']}\n")
        for row in rows:
            f.write(f"{row['id']}|{row['folder']}|{row['signer']}|{row['annotation']}\n")

    print(f'done.')
    print(f'  original rows   : {len(source_df)}')
    print(f'  npz files found : {len(npz_files)}')
    print(f'  augmented rows  : {len(rows)}')
    print(f'  total rows      : {len(source_df) + len(rows)}')
    print(f'  skipped         : {skipped}')
    print(f'  output csv      : {output_csv}')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='generate annotations CSV from an already-augmented NPZ directory',
    )
    p.add_argument('--aug_dir',    required=True, help='directory containing augmented NPZ files')
    p.add_argument('--source_csv', required=True, help='original annotations CSV (Phoenix format)')
    p.add_argument('--output_csv', required=True, help='path for the output annotations CSV')
    p.add_argument('--quiet',      action='store_true')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    build_augmented_csv(
        aug_dir    = Path(args.aug_dir),
        source_csv = Path(args.source_csv),
        output_csv = Path(args.output_csv),
        verbose    = not args.quiet,
    )