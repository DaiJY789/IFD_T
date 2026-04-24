import argparse
import base64
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from tqdm import tqdm


LABELS = ["authentic", "tampered", "ai_manipulated", "uncertain"]
AI_TYPES = ["ai_generated", "ai_eited"]


def normalize_endpoint(raw: str) -> str:
    endpoint = raw.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/chat/completions"
    return f"{endpoint}/v1/chat/completions"


def read_text(path: str) -> str:
    p = Path(path).expanduser().resolve()
    return p.read_text(encoding="utf-8").strip()


def normalize_label(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in {"authentic", "real", "re"}:
        return "authentic"
    if s in {"tampered", "tp", "tamper", "manipulated"}:
        return "tampered"
    if s in {"ai_manipulated", "ag", "aigc", "ai", "deepfake"}:
        return "ai_manipulated"
    if s in {"uncertain", "unknown"}:
        return "uncertain"
    return ""


def normalize_type(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in {"ai_generated", "generated", "aigc", "deepfake"}:
        return "ai_generated"
    if s in {"ai_eited", "ai_edited", "edited"}:
        return "ai_eited"
    return ""


def pick_first_type(v: Any) -> str:
    if isinstance(v, (list, tuple)) and v:
        return normalize_type(v[0])
    if hasattr(v, "tolist"):
        try:
            arr = v.tolist()
            if isinstance(arr, (list, tuple)) and arr:
                return normalize_type(arr[0])
        except Exception:
            pass
    if isinstance(v, str):
        return normalize_type(v)
    return ""


def load_eval(path: str, max_samples: int) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    df = pd.read_parquet(p)

    if "image_path" not in df.columns or "label" not in df.columns:
        raise ValueError(f"eval parquet missing required columns image_path/label: {p}")

    out = pd.DataFrame()
    out["image_path"] = df["image_path"].astype(str)
    out["gt_label"] = df["label"].apply(normalize_label)
    if "forgery_types" in df.columns:
        out["gt_type"] = df["forgery_types"].apply(pick_first_type)
    else:
        out["gt_type"] = ""

    if max_samples > 0:
        out = out.iloc[:max_samples].copy()
    return out.reset_index(drop=True)


def encode_image_data_url(image_path: str) -> str:
    p = Path(image_path).expanduser().resolve()
    suffix = p.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix, "image/jpeg")
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_json_obj(text: str) -> Optional[dict]:
    content = (text or "").strip()
    if not content:
        return None

    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def parse_prediction(raw_text: str) -> tuple[str, str, bool]:
    obj = extract_json_obj(raw_text)
    if not isinstance(obj, dict):
        return "", "", False

    pred_label = normalize_label(obj.get("verdict"))
    pred_type = ""
    ftypes = obj.get("forgery_types")
    if isinstance(ftypes, list) and ftypes:
        pred_type = normalize_type(ftypes[0])

    if pred_label == "ai_manipulated" and not pred_type:
        low = raw_text.lower()
        if "ai_eited" in low or "ai_edited" in low:
            pred_type = "ai_eited"
        elif "ai_generated" in low or "aigc" in low or "deepfake" in low:
            pred_type = "ai_generated"

    return pred_label, pred_type, True


def f1_for_label(y_true: list[str], y_pred: list[str], label: str) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_multiclass(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    per_label = {lb: f1_for_label(y_true, y_pred, lb) for lb in labels}
    macro_f1 = sum(m["f1"] for m in per_label.values()) / len(labels) if labels else 0.0

    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t not in cm:
            cm[t] = {x: 0 for x in labels}
        if p not in cm[t]:
            cm[t][p] = 0
        cm[t][p] += 1

    return {"accuracy": acc, "macro_f1": macro_f1, "per_label": per_label, "confusion_matrix": cm}


def post_chat(endpoint: str, payload: dict[str, Any], timeout: int, retries: int) -> dict[str, Any]:
    last_err = None
    for _ in range(retries):
        try:
            resp = requests.post(endpoint, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"request failed after {retries} retries: {last_err}")


def run(args: argparse.Namespace) -> None:
    eval_df = load_eval(args.eval_input, args.max_samples)
    endpoint = normalize_endpoint(args.chat_endpoint)

    system_prompt = read_text(args.system_prompt)
    user_template = read_text(args.user_template)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.jsonl"
    metrics_file = out_dir / "metrics.json"

    y_true = []
    y_pred = []
    ai_true = []
    ai_pred = []
    pairs: list[tuple[str, str, str, str]] = []

    ok = 0
    total_latency = 0
    start_ts = time.time()

    total_n = len(eval_df)
    with pred_file.open("w", encoding="utf-8") as wf:
        progress_iter = tqdm(
            eval_df.iterrows(),
            total=total_n,
            desc="Evaluating",
            dynamic_ncols=True,
            ncols=args.progress_width,
        )
        for i, row in progress_iter:
            image_path = str(row["image_path"])
            gt_label = str(row["gt_label"])
            gt_type = str(row["gt_type"])

            rec = {
                "index": int(i),
                "image_path": image_path,
                "gt_label": gt_label,
                "gt_type": gt_type,
                "pred_label": "",
                "pred_type": "",
                "request_ok": False,
                "json_ok": False,
                "error": "",
                "latency_ms": 0,
                "raw_output": "",
            }

            t0 = time.time()
            try:
                if not Path(image_path).expanduser().exists():
                    raise FileNotFoundError(f"image not found: {image_path}")

                user_text = user_template.replace("{{image_path}}", image_path)
                image_url = encode_image_data_url(image_path)

                payload = {
                    "model": args.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_text},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        },
                    ],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                }

                resp_obj = post_chat(endpoint, payload, args.timeout, args.retries)
                msg = (resp_obj.get("choices") or [{}])[0].get("message") or {}
                content = msg.get("content", "")
                if isinstance(content, list):
                    text = "\n".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
                else:
                    text = str(content)

                pred_label, pred_type, json_ok = parse_prediction(text)
                rec["pred_label"] = pred_label
                rec["pred_type"] = pred_type
                rec["json_ok"] = bool(json_ok)
                rec["request_ok"] = True
                rec["raw_output"] = text
                ok += 1
            except Exception as exc:  # noqa: BLE001
                rec["error"] = str(exc)

            rec["latency_ms"] = int((time.time() - t0) * 1000)
            total_latency += rec["latency_ms"]

            y_true.append(gt_label)
            y_pred.append(rec["pred_label"])
            pairs.append((gt_label, rec["pred_label"], gt_type, rec["pred_type"]))
            if gt_label == "ai_manipulated":
                ai_true.append(gt_type)
                ai_pred.append(rec["pred_type"])

            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done = i + 1

            if args.log_every > 0 and (done % args.log_every == 0 or done == total_n):
                tqdm.write(
                    f"[{done}/{total_n}] ok={rec['request_ok']} json={rec['json_ok']} "
                    f"pred={rec['pred_label']} gt={gt_label} t={rec['latency_ms']}ms"
                )

    label_metrics = evaluate_multiclass(y_true, y_pred, LABELS)
    ai_type_metrics = evaluate_multiclass(ai_true, ai_pred, AI_TYPES) if ai_true else {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "per_label": {},
        "confusion_matrix": {},
    }

    joint_ok = []
    for gt_l, pd_l, gt_t, pd_t in pairs:
        if gt_l == "ai_manipulated" and gt_t:
            joint_ok.append((gt_l == pd_l) and (gt_t == pd_t))
        else:
            joint_ok.append(gt_l == pd_l)

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eval_input": str(Path(args.eval_input).expanduser().resolve()),
        "num_samples": int(len(eval_df)),
        "request_ok": int(ok),
        "request_success_rate": ok / len(eval_df) if len(eval_df) else 0.0,
        "avg_latency_ms": total_latency / len(eval_df) if len(eval_df) else 0.0,
        "elapsed_seconds": round(time.time() - start_ts, 3),
        "chat_endpoint": endpoint,
        "model_name": args.model_name,
        "label_metrics": label_metrics,
        "ai_type_metrics": ai_type_metrics,
        "joint_accuracy": sum(joint_ok) / len(joint_ok) if joint_ok else 0.0,
        "outputs": {
            "predictions": str(pred_file),
            "metrics": str(metrics_file),
        },
    }

    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[DONE] vLLM baseline model evaluation finished")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baseline model on IFD_T eval.parquet via vLLM")
    p.add_argument(
        "--eval-input",
        "--eval-parquet",
        dest="eval_input",
        default="/ssd2/jydai/data/IFD_T/ifd_t_105k_v1/eval.parquet",
    )
    p.add_argument(
        "--chat-endpoint",
        "--base-url",
        dest="chat_endpoint",
        default="http://127.0.0.1:8010/",
    )
    p.add_argument("--model-name", default="Qwen3vl8b")
    p.add_argument(
        "--system-prompt",
        default="/data/home/yxzhou/jydai/work/IFD_T/prompt/system_prompt_baseline_no_tools.md",
    )
    p.add_argument(
        "--user-template",
        default="/data/home/yxzhou/jydai/work/IFD_T/prompt/user_prompt_template_baseline_no_tools.md",
    )
    p.add_argument("--output-dir", default="/ssd2/jydai/data/IFD_T/ifd_t_105k_v1/eval_vllm_baseline_qwen3vl8b")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--progress-width", type=int, default=120)
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
