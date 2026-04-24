import argparse
import json
import os
from typing import Any

import pandas as pd


TOOL_NAMES = (
    "spectrum_grid_analyzer",
    "facial_biological_auditor",
    "roi_extractor",
    "image_resizer",
    "visual_enhancer",
    "srm_filter",
    "noise_analyzer",
    "patchdense_scanner",
    "cfa_validator",
)


def _to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_native(tolist())
        except Exception:
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    return value


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _normalize_forgery_types(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        return [s]
    if isinstance(value, (list, tuple)):
        out = []
        for x in value:
            sx = str(x).strip()
            if sx:
                out.append(sx)
        return out
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        arr = tolist()
        if isinstance(arr, (list, tuple)):
            return _normalize_forgery_types(arr)
        return _normalize_forgery_types(str(arr))
    return []


def _build_tools_kwargs(image_path: str) -> dict[str, Any]:
    return {
        name: {
            "create_kwargs": {"image_input": image_path, "image_path": image_path},
            "execute_kwargs": {"image_input": image_path, "image_path": image_path},
        }
        for name in TOOL_NAMES
    }


def _build_one_row(
    row: dict[str, Any],
    idx: int,
    data_source: str,
    system_prompt: str,
    user_prompt_template: str,
    enable_tools_kwargs: bool,
) -> dict[str, Any]:
    image_path = str(row.get("image_path", "")).strip()
    if not image_path:
        raise ValueError(f"row {idx} missing image_path")

    label = str(row.get("label", "")).strip() or "uncertain"
    forgery_types = _normalize_forgery_types(row.get("forgery_types"))
    source_subset = str(row.get("source_subset", "")).strip()

    user_text = user_prompt_template.replace("{{image_path}}", image_path)
    prompt = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    extra_info: dict[str, Any] = {
        "index": idx,
        "original_image_path": image_path,
        "label": label,
        "forgery_types": forgery_types,
        "source_subset": source_subset,
    }

    if enable_tools_kwargs:
        extra_info["need_tools_kwargs"] = True
        extra_info["tools_kwargs"] = _build_tools_kwargs(image_path)

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "forgery_detection",
        "reward_model": {
            "style": "rule",
            "ground_truth": label,
        },
        "agent_name": "tool_agent",
        "extra_info": extra_info,
    }


def _rewrite_split(
    in_path: str,
    out_path: str,
    *,
    data_source: str,
    system_prompt: str,
    user_prompt_template: str,
    enable_tools_kwargs: bool,
    max_rows: int,
) -> int:
    if not os.path.exists(in_path):
        return 0

    df = pd.read_parquet(in_path)
    if max_rows > 0:
        df = df.iloc[:max_rows].copy()

    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        rows.append(
            _build_one_row(
                r.to_dict(),
                idx=i,
                data_source=data_source,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                enable_tools_kwargs=enable_tools_kwargs,
            )
        )

    out_df = pd.DataFrame(rows)
    out_df["prompt"] = out_df["prompt"].map(lambda x: json.dumps(x, ensure_ascii=False))
    out_df.to_parquet(out_path)
    return len(out_df)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build IFD_T tool-RL parquet dataset for verl training")
    parser.add_argument("--input-dir", default="/ssd2/jydai/data/IFD_T/ifd_t_105k_v1")
    parser.add_argument("--output-dir", default="/ssd2/jydai/data/IFD_T/ifd_t_105k_tool_rl_v1")
    parser.add_argument("--system-prompt-file", default="/data/home/yxzhou/jydai/work/IFD_T/prompt/system_prompt.md")
    parser.add_argument("--user-prompt-template-file", default="/data/home/yxzhou/jydai/work/IFD_T/prompt/user_prompt_template.md")
    parser.add_argument("--data-source", default="ifd_t_tool_rl_v1")
    parser.add_argument("--disable-tools-kwargs", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0, help="debug only; 0 means all rows")
    args = parser.parse_args()

    system_prompt = _load_text(args.system_prompt_file)
    user_prompt_template = _load_text(args.user_prompt_template_file)

    os.makedirs(args.output_dir, exist_ok=True)

    stats = {}
    for split in ("train", "test", "eval"):
        in_path = os.path.join(args.input_dir, f"{split}.parquet")
        out_path = os.path.join(args.output_dir, f"{split}.parquet")
        n = _rewrite_split(
            in_path,
            out_path,
            data_source=args.data_source,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            enable_tools_kwargs=not args.disable_tools_kwargs,
            max_rows=args.max_rows,
        )
        if n > 0:
            stats[split] = {"rows": n, "path": out_path}
            print(f"[OK] {split}: {out_path} ({n} rows)")

    eval_parquet = os.path.join(args.output_dir, "eval.parquet")
    if os.path.exists(eval_parquet):
        eval_df = pd.read_parquet(eval_parquet)
        eval_rows = []
        for _, row in eval_df.iterrows():
            d = row.to_dict()
            if isinstance(d.get("prompt"), str):
                d["prompt"] = json.loads(d["prompt"])
            eval_rows.append(_to_native(d))
        eval_json = os.path.join(args.output_dir, "eval.json")
        with open(eval_json, "w", encoding="utf-8") as f:
            json.dump(eval_rows, f, ensure_ascii=False, indent=2)
        stats["eval_json"] = {"rows": len(eval_rows), "path": eval_json}
        print(f"[OK] eval.json: {eval_json} ({len(eval_rows)} rows)")

    stats_path = os.path.join(args.output_dir, "build_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[OK] stats: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
