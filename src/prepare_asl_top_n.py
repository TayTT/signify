import argparse
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd


SPLIT_NAMES = ["train.csv", "val.csv", "test.csv"]
GLOSS_COL = "Gloss"
VIDEO_COL = "Video file"


def load_splits(splits_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for name in SPLIT_NAMES:
        p = splits_dir / name
        if p.exists():
            frames[name] = pd.read_csv(p)
        else:
            print(f"warning: {p} not found, skipping")
    return frames


def top_n_glosses(splits: dict[str, pd.DataFrame], n: int) -> list[str]:
    counter: Counter = Counter()
    for df in splits.values():
        if GLOSS_COL not in df.columns:
            raise ValueError(f"column '{GLOSS_COL}' not found, got: {list(df.columns)}")
        counter.update(df[GLOSS_COL].dropna().str.strip())
    return [gloss for gloss, _ in counter.most_common(n)]


def filter_split(df: pd.DataFrame, glosses: set[str]) -> pd.DataFrame:
    mask = df[GLOSS_COL].str.strip().isin(glosses)
    return df[mask].reset_index(drop=True)


def copy_videos(df_list: list[pd.DataFrame], src_videos: Path, dst_videos: Path) -> tuple[int, int]:
    dst_videos.mkdir(parents=True, exist_ok=True)
    filenames: set[str] = set()  # unique across all splits
    for df in df_list:
        filenames.update(df[VIDEO_COL].dropna().str.strip())

    ok = 0
    missing = 0
    for fname in sorted(filenames):
        src = src_videos / fname
        if not src.exists():
            print(f"  missing: {fname}")
            missing += 1
            continue
        shutil.copy2(src, dst_videos / fname)
        ok += 1
    return ok, missing


def main():
    parser = argparse.ArgumentParser(
        description="filter ASL Citizen dataset to top-N most frequent glosses"
    )
    parser.add_argument("input_dir", help="path to ASL_Citizen root (contains splits/ and videos/)")
    parser.add_argument("n", type=int, help="number of top glosses to keep")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    n = args.n

    if not input_dir.exists():
        raise SystemExit(f"input dir not found: {input_dir}")

    splits_dir = input_dir / "splits"
    videos_dir = input_dir / "videos"

    if not splits_dir.exists():
        raise SystemExit(f"splits/ not found under {input_dir}")
    if not videos_dir.exists():
        raise SystemExit(f"videos/ not found under {input_dir}")

    out_dir = input_dir.parent / f"{input_dir.name}top{n}"  # e.g. ASL_Citizentop50
    out_splits = out_dir / "splits"
    out_splits.mkdir(parents=True, exist_ok=True)

    print(f"input:  {input_dir}")
    print(f"output: {out_dir}")
    print(f"top-N:  {n}")

    splits = load_splits(splits_dir)
    if not splits:
        raise SystemExit("no split csvs found")

    glosses = top_n_glosses(splits, n)
    gloss_set = set(glosses)
    print(f"\ntop {n} glosses: {glosses[:10]}{'...' if n > 10 else ''}")

    filtered: dict[str, pd.DataFrame] = {}
    for name, df in splits.items():
        fdf = filter_split(df, gloss_set)
        filtered[name] = fdf
        out_path = out_splits / name
        fdf.to_csv(out_path, index=False)
        print(f"  {name}: {len(df)} -> {len(fdf)} rows  ->  {out_path}")

    print("\ncopying videos...")
    ok, missing = copy_videos(list(filtered.values()), videos_dir, out_dir / "videos")
    print(f"  copied: {ok}  missing: {missing}")
    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()