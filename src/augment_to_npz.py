#!/usr/bin/env python3
import json
import random
import argparse
import numpy as np
from pathlib import Path
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skeleton_augmentation import AugmentationConfig, augment_json
from preprocess_jsons import CORE_POSE_LANDMARKS
from json_to_npz import HAND_LANDMARKS, FACE_LANDMARKS


COPIES_PER_FILE = 9
CONTROL_EVERY   = 1000

# random param ranges
JITTER_SIGMA_MIN = 0.005
JITTER_SIGMA_MAX = 0.02
SCALE_MIN_LO     = 0.75   # lower bound for scale_min
SCALE_MIN_HI     = 0.95   # upper bound for scale_min
SCALE_MAX_LO     = 1.05   # lower bound for scale_max
SCALE_MAX_HI     = 1.25   # upper bound for scale_max
ROTATE_MAX_LO    = 5.0
ROTATE_MAX_HI    = 20.0

_AUG_POOL = ['flip', 'jitter', 'scale', 'rotate']


def _random_config(rng: random.Random) -> AugmentationConfig:
    k      = rng.randint(1, len(_AUG_POOL))  # at least one aug
    chosen = set(rng.sample(_AUG_POOL, k))
    return AugmentationConfig(
        flip         = 'flip'   in chosen,
        jitter       = 'jitter' in chosen,
        scale        = 'scale'  in chosen,
        rotate       = 'rotate' in chosen,
        jitter_sigma = rng.uniform(JITTER_SIGMA_MIN, JITTER_SIGMA_MAX) if 'jitter' in chosen else 0.01,
        scale_min    = rng.uniform(SCALE_MIN_LO, SCALE_MIN_HI)         if 'scale'  in chosen else 0.85,
        scale_max    = rng.uniform(SCALE_MAX_LO, SCALE_MAX_HI)         if 'scale'  in chosen else 1.15,
        rotate_max   = rng.uniform(ROTATE_MAX_LO, ROTATE_MAX_HI)       if 'rotate' in chosen else 10.0,
    )


def _data_to_npz(data: dict, out_path: Path):
    """in-memory equivalent of json_to_npz — avoids roundtripping through disk"""
    frames_data = data.get('frames', {})
    metadata    = data.get('metadata', {})

    if not frames_data:
        raise ValueError('no frames in data dict')

    sorted_keys = sorted(frames_data.keys(), key=lambda x: int(x))
    n_frames    = len(sorted_keys)

    frame_numbers   = np.zeros(n_frames, dtype=np.int32)
    hands           = np.zeros((n_frames, 2, HAND_LANDMARKS, 3), dtype=np.float32)
    hand_tracked    = np.zeros((n_frames, 2), dtype=bool)
    hand_confidence = np.zeros((n_frames, 2), dtype=np.float32)
    face            = np.zeros((n_frames, FACE_LANDMARKS, 3), dtype=np.float32)
    face_detected   = np.zeros(n_frames, dtype=bool)
    pose            = np.zeros((n_frames, len(CORE_POSE_LANDMARKS), 3), dtype=np.float32)

    for i, key in enumerate(sorted_keys):
        frame_numbers[i] = int(key)
        fd = frames_data[key]

        hands_raw = fd.get('hands', {})
        for h_idx, side in enumerate(['left_hand', 'right_hand']):
            hand_info = hands_raw.get(side)
            if not hand_info:
                continue
            if isinstance(hand_info, dict):
                lms  = hand_info.get('landmarks', [])
                conf = hand_info.get('confidence', 0.0)
            elif isinstance(hand_info, list):
                lms  = hand_info
                conf = 1.0
            else:
                continue
            if not lms:
                continue
            hand_tracked[i, h_idx]    = True
            hand_confidence[i, h_idx] = float(conf)
            for j, lm in enumerate(lms[:HAND_LANDMARKS]):
                if isinstance(lm, dict):
                    hands[i, h_idx, j] = [lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0)]

        face_raw = fd.get('face', {})
        face_lms = face_raw.get('all_landmarks', []) if isinstance(face_raw, dict) else []
        if face_lms:
            face_detected[i] = True
            for j, lm in enumerate(face_lms[:FACE_LANDMARKS]):
                if isinstance(lm, dict):
                    face[i, j] = [lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0)]

        pose_raw = fd.get('pose', {})
        for j, name in enumerate(CORE_POSE_LANDMARKS):
            lm = pose_raw.get(name)
            if lm and isinstance(lm, dict):
                pose[i, j] = [lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0)]

    fps    = float(metadata.get('fps', 25.0))
    width  = int(metadata.get('width', 0))
    height = int(metadata.get('height', 0))
    source = str(metadata.get('input_source', out_path.stem))

    np.savez_compressed(
        out_path,
        frame_numbers         = frame_numbers,
        hands                 = hands,
        hand_tracked          = hand_tracked,
        hand_confidence       = hand_confidence,
        face                  = face,
        face_detected         = face_detected,
        pose                  = pose,
        metadata_fps          = np.array([fps]),
        metadata_total_frames = np.array([n_frames]),
        metadata_width        = np.array([width]),
        metadata_height       = np.array([height]),
        metadata_input_source = np.array([source]),
    )


def augment_directory(
    input_dir:     Path,
    output_dir:    Path,
    control_dir:   Path,
    copies:        int  = COPIES_PER_FILE,
    seed:          int  = 42,
    control_every: int  = CONTROL_EVERY,
    verbose:       bool = True,
):
    json_files = sorted(input_dir.glob('*.json'))
    if not json_files:
        raise FileNotFoundError(f'no JSON files in {input_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)

    rng            = random.Random(seed)  # controls aug selection + params
    total_written  = 0
    total_errors   = 0
    global_counter = 0  # counts all produced outputs

    for file_idx, json_path in enumerate(json_files):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
        except Exception as e:
            print(f'[ERROR] reading {json_path.name}: {e}')
            total_errors += 1
            continue

        for copy_idx in range(copies):
            global_counter += 1

            cfg      = _random_config(rng)
            aug_seed = seed + file_idx * copies + copy_idx  # deterministic per (file, copy)

            try:
                augmented = augment_json(source_data, cfg, seed=aug_seed)
            except Exception as e:
                print(f'[ERROR] augmenting {json_path.name} copy {copy_idx}: {e}')
                total_errors += 1
                continue

            out_stem = f'{json_path.stem}_aug{copy_idx + 1:02d}_{cfg.tag()}'

            npz_path = output_dir / f'{out_stem}.npz'
            try:
                _data_to_npz(augmented, npz_path)
                total_written += 1
            except Exception as e:
                print(f'[ERROR] writing NPZ {out_stem}: {e}')
                total_errors += 1
                continue

            if global_counter % control_every == 0:  # every Nth output -> control sample
                ctrl_path = control_dir / f'{out_stem}.json'
                try:
                    with open(ctrl_path, 'w', encoding='utf-8') as f:
                        json.dump(augmented, f)
                    if verbose:
                        print(f'  [control #{global_counter // control_every}] {ctrl_path.name}')
                except Exception as e:
                    print(f'  [ERROR] saving control JSON {ctrl_path.name}: {e}')

        if verbose and (file_idx + 1) % 100 == 0:
            print(f'  {file_idx + 1}/{len(json_files)} source files done...')

    print()
    print(f'done.')
    print(f'  source files  : {len(json_files)}')
    print(f'  npz written   : {total_written}')
    print(f'  control jsons : {global_counter // control_every}')
    print(f'  errors        : {total_errors}')
    print(f'  output_dir    : {output_dir}')
    print(f'  control_dir   : {control_dir}')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='augment Phoenix JSON files to NPZ with random augmentation combos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'example:\n'
            '  python src/augment_to_npz.py'
            ' --input_dir ./data/phoenix_jsons/phoenix_train'
            ' --output_dir ./data/phoenix_npz/phoenix_train_aug'
            ' --control_dir ./data/aug_control_samples'
            ' --seed 42'
        ),
    )
    p.add_argument('--input_dir',     required=True,  help='source JSON directory')
    p.add_argument('--output_dir',    required=True,  help='output NPZ directory')
    p.add_argument('--control_dir',   required=True,  help='directory for JSON control samples')
    p.add_argument('--copies',        type=int, default=COPIES_PER_FILE,
                   help=f'augmented copies per source file (default: {COPIES_PER_FILE})')
    p.add_argument('--control_every', type=int, default=CONTROL_EVERY,
                   help=f'save one JSON control sample every N outputs (default: {CONTROL_EVERY})')
    p.add_argument('--seed',          type=int, default=42)
    p.add_argument('--quiet',         action='store_true')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    augment_directory(
        input_dir     = Path(args.input_dir),
        output_dir    = Path(args.output_dir),
        control_dir   = Path(args.control_dir),
        copies        = args.copies,
        seed          = args.seed,
        control_every = args.control_every,
        verbose       = not args.quiet,
    )
