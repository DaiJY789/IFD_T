import argparse
import json
from pathlib import Path

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p.resolve())
    files.sort()
    return files


def add_records(records: list[dict], files: list[Path], label: str, forgery_types: list[str], source_subset: str):
    for p in files:
        records.append(
            {
                "image_path": str(p),
                "label": label,
                "forgery_types": forgery_types,
                "source_subset": source_subset,
            }
        )


def build_dataframe(sample_root: Path) -> pd.DataFrame:
    records: list[dict] = []

    # AU -> authentic, forgery_types = []
    au_files = collect_images(sample_root / "AU")
    add_records(records, au_files, label="authentic", forgery_types=[], source_subset="AU")

    # TP -> tampered, forgery_types = []
    tp_files = collect_images(sample_root / "TP")
    add_records(records, tp_files, label="tampered", forgery_types=[], source_subset="TP")

    # AM subsets
    am_root = sample_root / "AM"
    am_generated_files = collect_images(am_root / "RedFace4000") + collect_images(am_root / "DFbench36000")
    am_edited_files = collect_images(am_root / "RedFace3*4000") + collect_images(am_root / "DFBench8000")

    add_records(
        records,
        am_generated_files,
        label="ai_manipulated",
        forgery_types=["ai_generated"],
        source_subset="AM:RedFace4000+DFbench36000",
    )
    add_records(
        records,
        am_edited_files,
        label="ai_manipulated",
        forgery_types=["ai_eited"],
        source_subset="AM:RedFace3*4000+DFBench8000",
    )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No images were collected. Please verify sample_root.")

    # Sanity checks expected by user statement.
    total = len(df)
    if total < 105000:
        raise RuntimeError(f"Insufficient images: got {total}, expected at least 105000")

    return df


def split_dataframe(df: pd.DataFrame, train_n: int, test_n: int, eval_n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_need = train_n + test_n + eval_n
    if len(df) < total_need:
        raise RuntimeError(f"Need {total_need} rows, but only {len(df)} available")

    # Stratify by primary label + subtype to preserve ai_generated/ai_eited mix.
    strat_key = df["label"].astype(str) + "||" + df["forgery_types"].astype(str)
    sampled = (
        df.assign(_strat_key=strat_key)
        .groupby("_strat_key", group_keys=False)
        .apply(lambda x: x.sample(frac=1.0, random_state=seed))
        .reset_index(drop=True)
    )

    # Global shuffle after stratified chunk shuffle.
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled = sampled.iloc[:total_need].copy()

    train_df = sampled.iloc[:train_n].drop(columns=["_strat_key"]) if "_strat_key" in sampled.columns else sampled.iloc[:train_n]
    test_df = sampled.iloc[train_n : train_n + test_n].drop(columns=["_strat_key"]) if "_strat_key" in sampled.columns else sampled.iloc[train_n : train_n + test_n]
    eval_df = sampled.iloc[train_n + test_n : total_need].drop(columns=["_strat_key"]) if "_strat_key" in sampled.columns else sampled.iloc[train_n + test_n : total_need]

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> dict:
    out: dict = {"total": int(len(df)), "label_counts": {}, "subtype_counts": {}}
    out["label_counts"] = {k: int(v) for k, v in df["label"].value_counts().to_dict().items()}

    subtype_series = df["forgery_types"].apply(lambda x: ",".join(x) if isinstance(x, list) else str(x))
    out["subtype_counts"] = {k: int(v) for k, v in subtype_series.value_counts().to_dict().items()}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", default="/ssd2/jydai/data/IFD_T/sample", type=str)
    parser.add_argument("--output-root", default="/ssd2/jydai/data/processed/IFD_T/ifd_t_105k_v1", type=str)
    parser.add_argument("--train-size", default=90000, type=int)
    parser.add_argument("--test-size", default=10000, type=int)
    parser.add_argument("--eval-size", default=5000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    sample_root = Path(args.sample_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(sample_root)
    train_df, test_df, eval_df = split_dataframe(
        df,
        train_n=args.train_size,
        test_n=args.test_size,
        eval_n=args.eval_size,
        seed=args.seed,
    )

    train_path = output_root / "train.parquet"
    test_path = output_root / "test.parquet"
    eval_path = output_root / "eval.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    eval_df.to_parquet(eval_path, index=False)

    metadata = {
        "sample_root": str(sample_root),
        "output_root": str(output_root),
        "split_sizes": {
            "train": int(len(train_df)),
            "test": int(len(test_df)),
            "eval": int(len(eval_df)),
        },
        "mapping": {
            "AU": {"label": "authentic", "forgery_types": []},
            "TP": {"label": "tampered", "forgery_types": []},
            "AM:RedFace4000+DFbench36000": {
                "label": "ai_manipulated",
                "forgery_types": ["ai_generated"],
            },
            "AM:RedFace3*4000+DFBench8000": {
                "label": "ai_manipulated",
                "forgery_types": ["ai_eited"],
            },
        },
        "stats": {
            "train": summarize(train_df),
            "test": summarize(test_df),
            "eval": summarize(eval_df),
        },
    }

    with (output_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("[OK] Dataset generated")
    print(f"[OK] train: {train_path}")
    print(f"[OK] test : {test_path}")
    print(f"[OK] eval : {eval_path}")
    print(f"[OK] meta : {output_root / 'metadata.json'}")


if __name__ == "__main__":
    main()
