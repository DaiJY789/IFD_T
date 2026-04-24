#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build anonymized eval dataset with random image filenames")
    p.add_argument("--input-parquet", required=True, help="Source eval parquet path")
    p.add_argument("--output-dir", required=True, help="Output directory for anonymized dataset")
    p.add_argument(
        "--image-col",
        default="image_path",
        help="Image path column name in parquet",
    )
    p.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to materialize anonymized files",
    )
    p.add_argument(
        "--filename-bytes",
        type=int,
        default=16,
        help="Random filename entropy in bytes (hex length is 2x)",
    )
    p.add_argument(
        "--fanout",
        type=int,
        default=2,
        help="Directory fanout levels using hex prefix, set 0 for flat layout",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwrite existing output parquet and mapping files",
    )
    return p.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def random_hex_name(n_bytes: int) -> str:
    return uuid.uuid4().hex if n_bytes == 16 else os.urandom(n_bytes).hex()


def materialize(src: Path, dst: Path, mode: str) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        return

    if mode == "symlink":
        dst.symlink_to(src)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    shutil.copy2(src, dst)


def build_dataset(args: argparse.Namespace) -> dict:
    src_parquet = Path(args.input_parquet).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()
    out_images = out_root / "images"
    out_parquet = out_root / "eval_anonymized.parquet"
    out_mapping = out_root / "mapping.csv"
    out_meta = out_root / "meta.json"

    if not src_parquet.exists():
        raise FileNotFoundError(f"input parquet not found: {src_parquet}")

    if out_parquet.exists() and not args.overwrite:
        raise FileExistsError(f"output parquet exists, use --overwrite: {out_parquet}")

    out_root.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src_parquet)
    if args.image_col not in df.columns:
        raise ValueError(f"missing image column: {args.image_col}")

    anonymized_paths = []
    mapping_rows = []
    used = set()
    missing = []

    for idx, raw_path in enumerate(df[args.image_col].astype(str).tolist()):
        src = Path(raw_path).expanduser().resolve()
        if not src.exists():
            missing.append(str(src))
            anonymized_paths.append("")
            continue

        ext = src.suffix.lower() if src.suffix else ".bin"
        for _ in range(20):
            stem = random_hex_name(args.filename_bytes)
            if stem not in used:
                used.add(stem)
                break
        else:
            raise RuntimeError("failed to generate unique random filename")

        if args.fanout > 0:
            parts = [stem[i * 2 : i * 2 + 2] for i in range(args.fanout)]
            rel = Path(*parts) / f"{stem}{ext}"
        else:
            rel = Path(f"{stem}{ext}")

        dst = out_images / rel
        materialize(src, dst, args.link_mode)

        anonymized_paths.append(str(dst))
        mapping_rows.append(
            {
                "index": idx,
                "original_path": str(src),
                "anonymized_path": str(dst),
                "anonymized_name": dst.name,
                "ext": ext,
            }
        )

    if missing:
        missing_preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"{len(missing)} source images missing, first entries:\n{missing_preview}"
        )

    out_df = df.copy()
    out_df[args.image_col] = anonymized_paths

    out_df.to_parquet(out_parquet, index=False)
    pd.DataFrame(mapping_rows).to_csv(out_mapping, index=False)

    meta = {
        "input_parquet": str(src_parquet),
        "output_parquet": str(out_parquet),
        "mapping_csv": str(out_mapping),
        "image_root": str(out_images),
        "num_samples": int(len(out_df)),
        "link_mode": args.link_mode,
        "fanout": int(args.fanout),
        "image_col": args.image_col,
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    args = parse_args()
    meta = build_dataset(args)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
