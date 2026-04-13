#!/usr/bin/env python3
"""
Skeleton-based data augmentation for Phoenix-format JSON landmark files.

Operates directly on landmark coordinate dicts so no video re-processing is needed.
All augmentations preserve the original JSON structure and missing-data flags.

Supported augmentations (combinable in any combination):
    jitter  - Gaussian noise on all coordinates  (simulates MediaPipe estimation noise)
    flip    - Horizontal mirror                   (simulates left-handed signers)
    scale   - Uniform spatial scaling             (simulates signer distance variation)
    rotate  - 2D rotation around centroid         (simulates camera angle variation)

Usage (CLI):
    python src/skeleton_augmentation.py
        --input_dir  ./data/phoenix_jsons/phoenix_train
        --output_dir ./data/phoenix_jsons/phoenix_train_aug
        --augmentations jitter flip scale rotate
        --jitter_sigma  0.01
        --scale_range   0.85 1.15
        --rotate_range  10.0
        --seed 42

Usage (API):
    from skeleton_augmentation import augment_json, AugmentationConfig

    cfg = AugmentationConfig(jitter=True, flip=True)
    augmented_data = augment_json(original_data, cfg)
"""

import json
import math
import copy
import random
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------- configuration ----------

@dataclass
class AugmentationConfig:
    jitter:       bool  = False
    flip:         bool  = False
    scale:        bool  = False
    rotate:       bool  = False

    jitter_sigma: float = 0.01           # std of gaussian noise on normalised coords
    scale_min:    float = 0.85
    scale_max:    float = 1.15
    rotate_max:   float = 10.0           # degrees, applied uniformly in [-max, +max]

    def tag(self) -> str:
        """short filename suffix encoding which augmentations are active"""
        parts = []
        if self.jitter: parts.append('jit')
        if self.flip:   parts.append('flp')
        if self.scale:  parts.append('scl')
        if self.rotate: parts.append('rot')
        return '_'.join(parts) if parts else 'noaug'

    def any_active(self) -> bool:
        return any([self.jitter, self.flip, self.scale, self.rotate])


# ---------- landmark helpers ----------

def _get_xy(lm: Dict) -> Tuple[float, float]:
    return lm.get('x', 0.0), lm.get('y', 0.0)


def _set_xy(lm: Dict, x: float, y: float) -> Dict:
    lm['x'] = x
    lm['y'] = y
    return lm


def _collect_all_landmarks(frame: Dict) -> List[Dict]:
    """
    Return flat list of all landmark dicts present in frame
    (hands + face + pose). Used to compute the body centroid.
    Missing hands return empty lists — centroid is computed from
    whatever is available.
    """
    lms = []
    hands = frame.get('hands', {})
    for side in ('left_hand', 'right_hand'):
        hand = hands.get(side)
        if hand and isinstance(hand, dict):
            lms.extend(hand.get('landmarks', []))

    face = frame.get('face', {})
    if face:
        lms.extend(face.get('all_landmarks', []))

    pose = frame.get('pose', {})
    if pose:
        lms.extend(pose.values())

    return [lm for lm in lms if isinstance(lm, dict) and 'x' in lm and 'y' in lm]


def _frame_centroid(frame: Dict) -> Tuple[float, float]:
    """mean x/y of all detected landmarks; falls back to (0.5, 0.5) if empty"""
    lms = _collect_all_landmarks(frame)
    if not lms:
        return 0.5, 0.5
    cx = sum(lm['x'] for lm in lms) / len(lms)
    cy = sum(lm['y'] for lm in lms) / len(lms)
    return cx, cy


def _apply_to_landmark_list(lms: List[Dict], fn) -> List[Dict]:
    return [fn(lm) for lm in lms]


def _apply_to_pose_dict(pose: Dict, fn) -> Dict:
    """pose landmarks are named, not a list"""
    return {name: fn(copy.copy(lm)) for name, lm in pose.items()}


# ---------- per-frame augmentation primitives ----------

def _jitter_landmark(lm: Dict, sigma: float, rng: np.random.Generator) -> Dict:
    lm = copy.copy(lm)
    lm['x'] = float(lm.get('x', 0.0) + rng.normal(0, sigma))
    lm['y'] = float(lm.get('y', 0.0) + rng.normal(0, sigma))
    # z is depth — jitter with smaller magnitude to avoid destroying relative depth
    lm['z'] = float(lm.get('z', 0.0) + rng.normal(0, sigma * 0.5))
    return lm


def _flip_landmark(lm: Dict) -> Dict:
    """mirror x around 0.5 (MediaPipe normalised space is [0,1])"""
    lm = copy.copy(lm)
    lm['x'] = float(1.0 - lm.get('x', 0.0))
    return lm


def _scale_landmark(lm: Dict, cx: float, cy: float, factor: float) -> Dict:
    """scale around centroid"""
    lm = copy.copy(lm)
    lm['x'] = float(cx + (lm.get('x', 0.0) - cx) * factor)
    lm['y'] = float(cy + (lm.get('y', 0.0) - cy) * factor)
    return lm


def _rotate_landmark(lm: Dict, cx: float, cy: float,
                     cos_a: float, sin_a: float) -> Dict:
    """2D rotation around centroid"""
    lm = copy.copy(lm)
    dx = lm.get('x', 0.0) - cx
    dy = lm.get('y', 0.0) - cy
    lm['x'] = float(cx + dx * cos_a - dy * sin_a)
    lm['y'] = float(cy + dx * sin_a + dy * cos_a)
    return lm


def _augment_frame(frame: Dict, cfg: AugmentationConfig,
                   rng: np.random.Generator,
                   scale_factor: float, cos_a: float, sin_a: float) -> Dict:
    """
    Apply all active augmentations to a single frame dict.
    scale_factor and cos_a/sin_a are pre-sampled per sequence (not per frame)
    so the transformation is consistent across the video.
    """
    frame = copy.deepcopy(frame)

    # centroid for scale and rotate (recalculated per-frame since landmarks vary)
    cx, cy = _frame_centroid(frame)

    def transform(lm):
        if not isinstance(lm, dict):
            return lm
        lm = copy.copy(lm)
        if cfg.jitter:
            lm = _jitter_landmark(lm, cfg.jitter_sigma, rng)
        if cfg.flip:
            lm = _flip_landmark(lm)
        if cfg.scale:
            lm = _scale_landmark(lm, cx, cy, scale_factor)
        if cfg.rotate:
            lm = _rotate_landmark(lm, cx, cy, cos_a, sin_a)
        return lm

    hands = frame.get('hands', {})
    for side in ('left_hand', 'right_hand'):
        hand = hands.get(side)
        if hand and isinstance(hand, dict) and 'landmarks' in hand:
            hands[side]['landmarks'] = [transform(lm) for lm in hand['landmarks']]

    face = frame.get('face', {})
    if face and 'all_landmarks' in face:
        face['all_landmarks'] = [transform(lm) for lm in face['all_landmarks']]

    pose = frame.get('pose', {})
    if pose:
        frame['pose'] = {name: transform(lm) for name, lm in pose.items()}

    # flip swaps handedness — swap hand entries so left/right remain anatomically correct
    if cfg.flip and 'hands' in frame:
        h = frame['hands']
        h['left_hand'], h['right_hand'] = h.get('right_hand'), h.get('left_hand')

    return frame


# ---------- sequence-level augmentation ----------

def augment_json(data: Dict, cfg: AugmentationConfig,
                 seed: Optional[int] = None) -> Dict:
    """
    Apply augmentations to a full JSON data dict (as loaded from file).
    Returns a new dict — original is not modified.

    Args:
        data: parsed JSON dict with 'frames' key
        cfg:  AugmentationConfig specifying which augmentations to apply
        seed: optional RNG seed for reproducibility

    Returns:
        augmented copy of data
    """
    if not cfg.any_active():
        return copy.deepcopy(data)

    rng = np.random.default_rng(seed)

    # sample sequence-level parameters once so transformation is consistent
    scale_factor = float(rng.uniform(cfg.scale_min, cfg.scale_max)) if cfg.scale else 1.0
    angle_deg    = float(rng.uniform(-cfg.rotate_max, cfg.rotate_max)) if cfg.rotate else 0.0
    angle_rad    = math.radians(angle_deg)
    cos_a        = math.cos(angle_rad)
    sin_a        = math.sin(angle_rad)

    result = copy.deepcopy(data)
    frames = result.get('frames', {})

    for key in frames:
        frames[key] = _augment_frame(
            frames[key], cfg, rng, scale_factor, cos_a, sin_a
        )

    # record augmentation params in metadata for traceability
    result.setdefault('augmentation', {})
    result['augmentation'].update({
        'jitter':       cfg.jitter,
        'flip':         cfg.flip,
        'scale':        cfg.scale,
        'rotate':       cfg.rotate,
        'jitter_sigma': cfg.jitter_sigma if cfg.jitter else None,
        'scale_factor': round(scale_factor, 4) if cfg.scale else None,
        'rotate_deg':   round(angle_deg, 4) if cfg.rotate else None,
    })

    return result


# ---------- file I/O ----------

def augmented_filename(original_path: Path, cfg: AugmentationConfig) -> str:
    """
    Build output filename: <original_stem>_aug_<tag>.json
    e.g. 01April_2010_Thursday_heute_default-1_aug_jit_flp.json
    """
    return f'{original_path.stem}_aug_{cfg.tag()}.json'


def process_file(json_path: Path, output_dir: Path, cfg: AugmentationConfig,
                 seed: Optional[int] = None) -> Path:
    """
    Load a JSON file, augment it, write the result to output_dir.
    Returns the path of the written file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    augmented = augment_json(data, cfg, seed=seed)

    out_name = augmented_filename(json_path, cfg)
    out_path = output_dir / out_name

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(augmented, f)  # no indent to keep file size down

    return out_path


def process_directory(input_dir: Path, output_dir: Path, cfg: AugmentationConfig,
                      seed: Optional[int] = None, verbose: bool = True) -> List[Path]:
    """
    Augment all JSON files in input_dir and write results to output_dir.
    Returns list of written paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(input_dir.glob('*.json'))

    if not json_files:
        raise FileNotFoundError(f'no JSON files found in {input_dir}')

    written = []
    for i, jf in enumerate(json_files):
        # derive per-file seed from global seed so results are reproducible
        # but each file still gets a unique transformation
        file_seed = (seed + i) if seed is not None else None
        try:
            out = process_file(jf, output_dir, cfg, seed=file_seed)
            written.append(out)
            if verbose:
                print(f'[{i+1}/{len(json_files)}] {jf.name} -> {out.name}')
        except Exception as e:
            print(f'ERROR processing {jf.name}: {e}')

    print(f'\ndone. {len(written)}/{len(json_files)} files written to {output_dir}')
    return written


# ---------- CLI ----------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='apply skeleton augmentation to Phoenix-format JSON landmark files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'examples:\n'
            '  python src/skeleton_augmentation.py \\\n'
            '      --input_dir  ./data/phoenix_jsons/phoenix_train \\\n'
            '      --output_dir ./data/phoenix_jsons/phoenix_train_aug \\\n'
            '      --augmentations jitter flip\n\n'
            '  python src/skeleton_augmentation.py \\\n'
            '      --input_dir  ./data/phoenix_jsons/phoenix_train \\\n'
            '      --output_dir ./data/phoenix_jsons/phoenix_train_aug \\\n'
            '      --augmentations jitter scale rotate \\\n'
            '      --jitter_sigma 0.015 --scale_range 0.9 1.1 --rotate_range 8.0 \\\n'
            '      --seed 42'
        )
    )
    p.add_argument('--input_dir',  required=True,
                   help='directory containing source JSON files')
    p.add_argument('--output_dir', required=True,
                   help='directory to write augmented JSON files')
    p.add_argument('--augmentations', nargs='+',
                   choices=['jitter', 'flip', 'scale', 'rotate'],
                   required=True,
                   help='one or more augmentations to apply')
    p.add_argument('--jitter_sigma', type=float, default=0.01,
                   help='gaussian noise std for jitter (default: 0.01)')
    p.add_argument('--scale_range', type=float, nargs=2, default=[0.85, 1.15],
                   metavar=('MIN', 'MAX'),
                   help='scale factor range (default: 0.85 1.15)')
    p.add_argument('--rotate_range', type=float, default=10.0,
                   help='max rotation in degrees, applied as +/- (default: 10.0)')
    p.add_argument('--seed', type=int, default=None,
                   help='random seed for reproducibility')
    p.add_argument('--quiet', action='store_true',
                   help='suppress per-file output')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()

    cfg = AugmentationConfig(
        jitter       = 'jitter' in args.augmentations,
        flip         = 'flip'   in args.augmentations,
        scale        = 'scale'  in args.augmentations,
        rotate       = 'rotate' in args.augmentations,
        jitter_sigma = args.jitter_sigma,
        scale_min    = args.scale_range[0],
        scale_max    = args.scale_range[1],
        rotate_max   = args.rotate_range,
    )

    print(f'augmentations : {", ".join(args.augmentations)}')
    print(f'output tag    : _aug_{cfg.tag()}')
    if cfg.jitter: print(f'  jitter_sigma: {cfg.jitter_sigma}')
    if cfg.scale:  print(f'  scale_range : {cfg.scale_min} - {cfg.scale_max}')
    if cfg.rotate: print(f'  rotate_range: +/- {cfg.rotate_max} deg')
    print()

    process_directory(
        input_dir  = Path(args.input_dir),
        output_dir = Path(args.output_dir),
        cfg        = cfg,
        seed       = args.seed,
        verbose    = not args.quiet,
    )
