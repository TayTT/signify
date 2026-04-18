#!/usr/bin/env python3
"""
Convert Phoenix-format JSON landmark files to NPZ format.

NPZ files are significantly smaller and load faster during training.
The NPZ structure matches exactly what SignLanguagePreprocessor._load_npz_data expects,
so converted files are a drop-in replacement for JSON files in the training pipeline.

Usage:
    # convert a whole directory in-place (keeps JSONs alongside)
    python src/json_to_npz.py --input_dir ./data/phoenix_jsons/phoenix_train_aug

    # convert to a separate output directory
    python src/json_to_npz.py --input_dir ./data/phoenix_jsons/phoenix_train_aug
                               --output_dir ./data/phoenix_npz/phoenix_train_aug

    # delete source JSONs after successful conversion
    python src/json_to_npz.py --input_dir ./data/phoenix_jsons/phoenix_train_aug
                               --delete-json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_jsons import CORE_POSE_LANDMARKS

HAND_LANDMARKS = 21
FACE_LANDMARKS = 468


# ---------- conversion ----------

def json_to_npz(json_path: Path, out_path: Path):
    """
    Read a Phoenix JSON landmark file and write an equivalent NPZ file.
    Arrays match the layout expected by SignLanguagePreprocessor._load_npz_data.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frames_data = data.get('frames', {})
    metadata    = data.get('metadata', {})

    if not frames_data:
        raise ValueError(f'no frames in {json_path.name}')

    sorted_keys = sorted(frames_data.keys(), key=lambda x: int(x))
    n_frames    = len(sorted_keys)

    # allocate arrays
    frame_numbers    = np.zeros(n_frames, dtype=np.int32)
    hands            = np.zeros((n_frames, 2, HAND_LANDMARKS, 3), dtype=np.float32)
    hand_tracked     = np.zeros((n_frames, 2), dtype=bool)
    hand_confidence  = np.zeros((n_frames, 2), dtype=np.float32)
    face             = np.zeros((n_frames, FACE_LANDMARKS, 3), dtype=np.float32)
    face_detected    = np.zeros(n_frames, dtype=bool)
    pose             = np.zeros((n_frames, len(CORE_POSE_LANDMARKS), 3), dtype=np.float32)

    for i, key in enumerate(sorted_keys):
        frame_numbers[i] = int(key)
        fd = frames_data[key]

        # hands
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
                    hands[i, h_idx, j] = [lm.get('x', 0.0),
                                          lm.get('y', 0.0),
                                          lm.get('z', 0.0)]

        # face
        face_raw = fd.get('face', {})
        face_lms = face_raw.get('all_landmarks', []) if isinstance(face_raw, dict) else []
        if face_lms:
            face_detected[i] = True
            for j, lm in enumerate(face_lms[:FACE_LANDMARKS]):
                if isinstance(lm, dict):
                    face[i, j] = [lm.get('x', 0.0),
                                  lm.get('y', 0.0),
                                  lm.get('z', 0.0)]

        # pose
        pose_raw = fd.get('pose', {})
        for j, name in enumerate(CORE_POSE_LANDMARKS):
            lm = pose_raw.get(name)
            if lm and isinstance(lm, dict):
                pose[i, j] = [lm.get('x', 0.0),
                               lm.get('y', 0.0),
                               lm.get('z', 0.0)]

    fps    = float(metadata.get('fps', 25.0))
    width  = int(metadata.get('width',  0))
    height = int(metadata.get('height', 0))
    source = str(metadata.get('input_source', json_path.stem))

    np.savez_compressed(
        out_path,
        frame_numbers          = frame_numbers,
        hands                  = hands,
        hand_tracked           = hand_tracked,
        hand_confidence        = hand_confidence,
        face                   = face,
        face_detected          = face_detected,
        pose                   = pose,
        metadata_fps           = np.array([fps]),
        metadata_total_frames  = np.array([n_frames]),
        metadata_width         = np.array([width]),
        metadata_height        = np.array([height]),
        metadata_input_source  = np.array([source]),
    )


# ---------- batch conversion ----------

def convert_directory(input_dir: Path, output_dir: Path,
                      delete_json: bool = False, verbose: bool = True):
    json_files = sorted(input_dir.glob('*.json'))
    if not json_files:
        raise FileNotFoundError(f'no JSON files found in {input_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    errors = 0
    for i, jf in enumerate(json_files):
        out = output_dir / (jf.stem + '.npz')
        try:
            json_to_npz(jf, out)
            ok += 1
            if delete_json:
                jf.unlink()
            if verbose and (i + 1) % 200 == 0:
                print(f'  {i + 1}/{len(json_files)} converted...')
        except Exception as e:
            print(f'  [ERROR] {jf.name}: {e}')
            errors += 1

    print(f'\ndone.  converted: {ok}  errors: {errors}  output: {output_dir}')


# ---------- CLI ----------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='convert Phoenix JSON landmark files to NPZ format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--input_dir',  required=True,
                   help='directory containing source JSON files')
    p.add_argument('--output_dir', default=None,
                   help='output directory for NPZ files (default: same as input_dir)')
    p.add_argument('--delete-json', action='store_true', default=False,
                   help='delete source JSON after successful conversion')
    p.add_argument('--quiet', action='store_true',
                   help='suppress progress output')
    return p


if __name__ == '__main__':
    args   = _build_parser().parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else in_dir
    convert_directory(in_dir, out_dir,
                      delete_json=args.delete_json,
                      verbose=not args.quiet)