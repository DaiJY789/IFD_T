#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Any


WATCH_KEYS = ("json_ok", "schema_ok", "semantic_ok", "acc")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor rollout JSON quality from verl metrics jsonl")
    p.add_argument("--metrics-jsonl", required=True, help="Path to train_metrics.jsonl")
    p.add_argument("--interval", type=int, default=30, help="Print interval in seconds")
    p.add_argument("--follow", action="store_true", help="Follow file growth")
    p.add_argument("--max-lines", type=int, default=0, help="Stop after reading N lines; 0 means no limit")
    return p.parse_args()


def as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if v in (0, 1):
            return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):  # noqa: PLR2004
            return True
        if s in ("false", "0", "no"):  # noqa: PLR2004
            return False
    return None


def collect_flags(obj: Any, out: dict[str, list[bool]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in WATCH_KEYS:
                b = as_bool(v)
                if b is not None:
                    out[k].append(b)
            collect_flags(v, out)
    elif isinstance(obj, list):
        for x in obj:
            collect_flags(x, out)


def fmt_ratio(vals: list[bool]) -> str:
    if not vals:
        return "n/a"
    ok = sum(1 for x in vals if x)
    total = len(vals)
    return f"{ok}/{total}={ok / total:.4f}"


def print_snapshot(flag_store: dict[str, list[bool]], lines_seen: int) -> None:
    print(
        "[MON] lines=",
        lines_seen,
        " json_ok=",
        fmt_ratio(flag_store["json_ok"]),
        " schema_ok=",
        fmt_ratio(flag_store["schema_ok"]),
        " semantic_ok=",
        fmt_ratio(flag_store["semantic_ok"]),
        " acc=",
        fmt_ratio(flag_store["acc"]),
        sep="",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    path = Path(args.metrics_jsonl).expanduser().resolve()

    if not path.exists():
        print(f"[ERR] metrics file not found: {path}")
        return 2

    flags = {k: [] for k in WATCH_KEYS}
    lines_seen = 0
    last_print_ts = 0.0

    with path.open("r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                if args.follow:
                    now = time.time()
                    if now - last_print_ts >= args.interval:
                        print_snapshot(flags, lines_seen)
                        last_print_ts = now
                    time.sleep(1)
                    continue
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            collect_flags(obj, flags)
            lines_seen += 1

            now = time.time()
            if now - last_print_ts >= args.interval:
                print_snapshot(flags, lines_seen)
                last_print_ts = now

            if args.max_lines > 0 and lines_seen >= args.max_lines:
                break

    print_snapshot(flags, lines_seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
