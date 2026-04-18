#!/usr/bin/env python3
"""
Vocab collision / gap diagnostic.

Usage:
    python check_vocab.py models_stash\ctc-phoenix-aug\lstm_sign2gloss.pth
"""

import sys
import pickle
import torch
from pathlib import Path
from collections import defaultdict


def check_vocab(vocab: dict, source: str):
    print(f"\n--- {source} ---")
    print(f"  forward entries (gloss->id) : {len(vocab)}")

    reverse = {}
    collisions = defaultdict(list)  # id -> [gloss, gloss, ...]

    for gloss, idx in vocab.items():
        if idx in reverse:
            collisions[idx].append(gloss)
            collisions[idx].append(reverse[idx]) if reverse[idx] not in collisions[idx] else None
        else:
            reverse[idx] = gloss

    print(f"  reverse entries (id->gloss) : {len(reverse)}")
    print(f"  collision ids              : {len(collisions)}")

    if collisions:
        print(f"\n  [!] colliding ids (multiple glosses share the same id):")
        for idx in sorted(collisions)[:20]:
            print(f"      id {idx:4d} -> {collisions[idx]}")
        if len(collisions) > 20:
            print(f"      ... and {len(collisions) - 20} more")

    # find gaps in id range
    all_ids = set(vocab.values())
    max_id = max(all_ids) if all_ids else 0
    expected = set(range(max_id + 1))
    gaps = sorted(expected - all_ids)

    print(f"\n  id range     : 0 - {max_id}")
    print(f"  gap count    : {len(gaps)}")
    if gaps:
        print(f"  first 20 gaps: {gaps[:20]}")

    # sample of ids that decode to unk in the test output
    suspect_ids = [517, 1225, 863, 959, 1003, 893, 508, 843, 1016,
                   869, 708, 883, 939, 874, 822, 500, 1089, 1090,
                   1123, 683, 1024, 871]
    print(f"\n  checking ids seen as <UNK_ID_N> in test output:")
    for sid in suspect_ids:
        gloss = reverse.get(sid, "<NOT IN REVERSE MAP>")
        fwd_check = [g for g, i in vocab.items() if i == sid]
        print(f"      id {sid:4d} -> reverse: {gloss:30s}  forward matches: {fwd_check}")


def main(checkpoint_path: str):
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"file not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    print(f"checkpoint keys: {list(checkpoint.keys())}")

    found = False

    if 'gloss_to_idx' in checkpoint:
        check_vocab(checkpoint['gloss_to_idx'], "checkpoint['gloss_to_idx']")
        found = True

    if 'vocab' in checkpoint:
        v = checkpoint['vocab']
        if isinstance(v, dict):
            check_vocab(v, "checkpoint['vocab']")
            found = True

    vocab_pkl = path.parent / 'vocab.pkl'
    if vocab_pkl.exists():
        with open(vocab_pkl, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            if 'gloss_to_idx' in data:
                check_vocab(data['gloss_to_idx'], "vocab.pkl['gloss_to_idx']")
                found = True
            elif 'vocab' in data:
                check_vocab(data['vocab'], "vocab.pkl['vocab']")
                found = True
            else:
                # might be the vocab dict itself
                check_vocab(data, "vocab.pkl (flat dict)")
                found = True

    if not found:
        print("no vocab found in checkpoint or vocab.pkl -- check checkpoint keys above")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python check_vocab.py <path_to_checkpoint.pth>")
        sys.exit(1)
    main(sys.argv[1])