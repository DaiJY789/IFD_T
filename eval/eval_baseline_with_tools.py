import argparse
import asyncio
import base64
import csv
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
WORK_ROOT = REPO_ROOT / "work"
IFD_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = REPO_ROOT / "verl"

for p in (REPO_ROOT, WORK_ROOT, IFD_ROOT, VERL_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from verl.tools.utils.tool_registry import initialize_tools_from_config


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
    return Path(path).expanduser().resolve().read_text(encoding="utf-8").strip()


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
    df = pd.read_parquet(Path(path).expanduser().resolve())
    if "image_path" not in df.columns or "label" not in df.columns:
        raise ValueError("eval parquet missing required columns image_path/label")

    out = pd.DataFrame()
    out["image_path"] = df["image_path"].astype(str)
    out["gt_label"] = df["label"].apply(normalize_label)
    out["gt_type"] = df["forgery_types"].apply(pick_first_type) if "forgery_types" in df.columns else ""

    if max_samples > 0:
        out = out.iloc[:max_samples].copy()
    return out.reset_index(drop=True)


def encode_image_data_url(image_path: str) -> str:
    p = Path(image_path).expanduser().resolve()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(p.suffix.lower(), "image/jpeg")
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

    # Try fenced code blocks first (```json ... ```), which are common in LLM output.
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE):
        block = match.group(1).strip()
        if not block:
            continue
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        inner_match = re.search(r"\{[\s\S]*\}", block)
        if not inner_match:
            continue
        try:
            obj = json.loads(inner_match.group(0))
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


def infer_label_from_keywords(raw_text: str) -> str:
    low = (raw_text or "").lower()
    if not low:
        return ""

    uncertain_kw = ["uncertain", "unknown", "inconclusive", "cannot determine", "not sure", "不确定", "无法判断"]
    ai_kw = [
        "ai_manipulated",
        "ai manipulated",
        "ai-generated",
        "ai generated",
        "aigc",
        "deepfake",
        "synthetic",
        "ai_edited",
        "ai edited",
        "ai生成",
        "ai编辑",
    ]
    tampered_kw = ["tampered", "manipulated", "forged", "forgery", "doctored", "splice", "copy-move", "inpaint", "篡改", "伪造"]
    authentic_kw = ["authentic", "real", "pristine", "untampered", "original", "genuine", "真实", "原图"]

    if any(k in low for k in uncertain_kw):
        return "uncertain"
    if any(k in low for k in ai_kw):
        return "ai_manipulated"
    if any(k in low for k in tampered_kw):
        return "tampered"
    if any(k in low for k in authentic_kw):
        return "authentic"
    return ""


def infer_type_from_keywords(raw_text: str) -> str:
    low = (raw_text or "").lower()
    if not low:
        return ""

    edited_kw = ["ai_eited", "ai_edited", "ai edited", "edited by ai", "局部编辑", "ai编辑"]
    generated_kw = ["ai_generated", "ai generated", "generated by ai", "aigc", "deepfake", "synthetic", "ai生成"]

    if any(k in low for k in edited_kw):
        return "ai_eited"
    if any(k in low for k in generated_kw):
        return "ai_generated"
    return ""


def parse_prediction(raw_text: str) -> tuple[str, str, bool]:
    obj = extract_json_obj(raw_text)
    pred_label = ""
    pred_type = ""

    if isinstance(obj, dict):
        # Accept multiple common key variants from model outputs.
        for key in ("verdict", "label", "pred_label", "prediction", "class", "result", "category"):
            if key in obj:
                pred_label = normalize_label(obj.get(key))
                if pred_label:
                    break

        for key in ("forgery_types", "type", "subtype", "manipulation_type"):
            if key not in obj:
                continue
            ftypes = obj.get(key)
            if isinstance(ftypes, list) and ftypes:
                pred_type = normalize_type(ftypes[0])
            else:
                pred_type = normalize_type(ftypes)
            if pred_type:
                break

        if not pred_label and pred_type:
            pred_label = "ai_manipulated"

        if pred_label == "ai_manipulated" and not pred_type:
            pred_type = infer_type_from_keywords(raw_text)

        if pred_label:
            return pred_label, pred_type, True

    # Fallback when JSON is missing/invalid: infer from free text.
    pred_label = infer_label_from_keywords(raw_text)
    if pred_label == "ai_manipulated":
        pred_type = infer_type_from_keywords(raw_text)
    return pred_label, pred_type, False


def extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
        return "\n".join(texts).strip()
    return str(content).strip()


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


def to_tool_schema_list(tools: list[Any]) -> list[dict[str, Any]]:
    schemas = []
    for tool in tools:
        schemas.append(tool.tool_schema.model_dump(exclude_none=True, exclude_unset=True))
    return schemas


async def execute_tool_once(tool: Any, image_path: str, parameters: dict[str, Any]) -> str:
    instance_id = None
    try:
        instance_id, _ = await tool.create(image_input=image_path, image_path=image_path)
        response, _, _ = await tool.execute(instance_id=instance_id, parameters=parameters)
        if hasattr(response, "text"):
            return str(response.text)
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)
    finally:
        if instance_id is not None:
            try:
                await tool.release(instance_id=instance_id)
            except Exception:
                pass


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


def run(args: argparse.Namespace) -> None:
    eval_df = load_eval(args.eval_input, args.max_samples)
    endpoint = normalize_endpoint(args.chat_endpoint)
    system_prompt = read_text(args.system_prompt)
    user_template = read_text(args.user_template)

    tools = initialize_tools_from_config(args.tool_config)
    if args.tool_index >= 0:
        if args.tool_index >= len(tools):
            raise ValueError(f"tool_index out of range: {args.tool_index}, available=[0, {len(tools)-1}]")
        tools = [tools[args.tool_index]]

    tool_map = {tool.name: tool for tool in tools}
    tool_schema_list = to_tool_schema_list(tools)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.jsonl"
    metrics_file = out_dir / "metrics.json"
    progress_file = out_dir / "progress_events.jsonl"
    viz_csv_file = out_dir / "predictions_for_viz.csv"
    run_log_file = out_dir / "run.log"

    y_true: list[str] = []
    y_pred: list[str] = []
    ai_true: list[str] = []
    ai_pred: list[str] = []
    pairs: list[tuple[str, str, str, str]] = []

    ok = 0
    json_ok_count = 0
    json_fail_count = 0
    total_latency = 0
    total_tool_calls = 0
    tool_call_counter: Counter[str] = Counter()
    start_ts = time.time()

    with (
        pred_file.open("w", encoding="utf-8") as wf,
        progress_file.open("w", encoding="utf-8") as pf,
        viz_csv_file.open("w", encoding="utf-8", newline="") as vf,
        run_log_file.open("w", encoding="utf-8") as lf,
    ):
        csv_writer = csv.DictWriter(
            vf,
            fieldnames=[
                "index",
                "image_path",
                "gt_label",
                "gt_type",
                "pred_label",
                "pred_type",
                "request_ok",
                "json_ok",
                "tool_calls",
                "tool_call_rounds",
                "latency_ms",
                "error",
            ],
        )
        csv_writer.writeheader()
        lf.write(f"[{datetime.now().isoformat(timespec='seconds')}] start evaluation\n")
        lf.flush()

        progress = tqdm(eval_df.iterrows(), total=len(eval_df), desc="Eval+Tools", dynamic_ncols=True)
        for i, row in progress:
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
                "tool_calls": 0,
                "tool_call_rounds": 0,
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

                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ]

                final_text = ""
                ended_with_tool_calls = False
                for _ in range(args.max_tool_rounds + 1):
                    payload = {
                        "model": args.model_name,
                        "messages": messages,
                        "tools": tool_schema_list,
                        "tool_choice": "auto",
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "max_completion_tokens": args.max_completion_tokens,
                    }
                    resp_obj = post_chat(endpoint, payload, args.timeout, args.retries)
                    msg = (resp_obj.get("choices") or [{}])[0].get("message") or {}

                    content = msg.get("content", "")
                    final_text = extract_text_content(content)
                    tool_calls = msg.get("tool_calls") or []

                    if not tool_calls:
                        ended_with_tool_calls = False
                        messages.append({"role": "assistant", "content": content})
                        break

                    ended_with_tool_calls = True
                    rec["tool_call_rounds"] += 1
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": tool_calls,
                        }
                    )

                    for tool_call in tool_calls:
                        fn = (tool_call.get("function") or {}).get("name", "")
                        args_text = (tool_call.get("function") or {}).get("arguments", "{}")
                        call_id = tool_call.get("id") or f"call_{i}_{rec['tool_calls']}"

                        tool = tool_map.get(fn)
                        if tool is None:
                            tool_output = json.dumps({"success": False, "error": f"Unknown tool: {fn}"}, ensure_ascii=False)
                        else:
                            try:
                                params = json.loads(args_text) if isinstance(args_text, str) else {}
                                if not isinstance(params, dict):
                                    params = {}
                            except Exception:
                                params = {}

                            if "image_input" not in params and "image_path" not in params:
                                params["image_input"] = str(Path(image_path).expanduser().resolve())

                            tool_output = asyncio.run(execute_tool_once(tool, str(Path(image_path).expanduser().resolve()), params))

                        rec["tool_calls"] += 1
                        tool_call_counter[fn] += 1
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": fn,
                                "content": tool_output,
                            }
                        )

                # If we hit tool-call budget and the model still asked for tools,
                # force one final text-only answer turn.
                if ended_with_tool_calls:
                    finalize_prompts = [
                        (
                            "工具调用预算已用尽。现在禁止再调用任何工具。"
                            "请直接输出最终结果，并且只输出一个合法 JSON 对象，"
                            "不要包含 <tool_call>、代码块、解释文字或额外前后缀。"
                        ),
                        (
                            "你上一条没有按格式输出。再次强调："
                            "禁止任何工具调用相关文本，只返回一个 JSON 对象本体。"
                        ),
                    ]

                    for finalize_prompt in finalize_prompts:
                        final_payload = {
                            "model": args.model_name,
                            "messages": messages + [{"role": "user", "content": finalize_prompt}],
                            "tools": tool_schema_list,
                            "tool_choice": "none",
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens,
                            "max_completion_tokens": args.max_completion_tokens,
                        }
                        final_resp_obj = post_chat(endpoint, final_payload, args.timeout, args.retries)
                        final_msg = (final_resp_obj.get("choices") or [{}])[0].get("message") or {}
                        final_content = final_msg.get("content", "")
                        final_text = extract_text_content(final_content)

                        if extract_json_obj(final_text) is not None:
                            break

                pred_label, pred_type, json_ok = parse_prediction(final_text)

                rec["pred_label"] = pred_label
                rec["pred_type"] = pred_type
                rec["json_ok"] = bool(json_ok)
                rec["request_ok"] = True
                rec["raw_output"] = final_text
                ok += 1
            except Exception as exc:  # noqa: BLE001
                rec["error"] = str(exc)

            rec["latency_ms"] = int((time.time() - t0) * 1000)
            total_latency += rec["latency_ms"]
            total_tool_calls += int(rec["tool_calls"])
            if rec["json_ok"]:
                json_ok_count += 1
            else:
                json_fail_count += 1

            y_true.append(gt_label)
            y_pred.append(rec["pred_label"])
            pairs.append((gt_label, rec["pred_label"], gt_type, rec["pred_type"]))
            if gt_label == "ai_manipulated":
                ai_true.append(gt_type)
                ai_pred.append(rec["pred_type"])

            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pf.write(
                json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "index": int(i),
                        "done": int(i + 1),
                        "total": int(len(eval_df)),
                        "gt_label": gt_label,
                        "pred_label": rec["pred_label"],
                        "json_ok": bool(rec["json_ok"]),
                        "request_ok": bool(rec["request_ok"]),
                        "tool_calls": int(rec["tool_calls"]),
                        "latency_ms": int(rec["latency_ms"]),
                        "error": rec["error"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            csv_writer.writerow(
                {
                    "index": int(i),
                    "image_path": image_path,
                    "gt_label": gt_label,
                    "gt_type": gt_type,
                    "pred_label": rec["pred_label"],
                    "pred_type": rec["pred_type"],
                    "request_ok": rec["request_ok"],
                    "json_ok": rec["json_ok"],
                    "tool_calls": rec["tool_calls"],
                    "tool_call_rounds": rec["tool_call_rounds"],
                    "latency_ms": rec["latency_ms"],
                    "error": rec["error"],
                }
            )
            pf.flush()
            vf.flush()

            done = i + 1
            if args.log_every > 0 and (done % args.log_every == 0 or done == len(eval_df)):
                msg = (
                    f"[{done}/{len(eval_df)}] ok={rec['request_ok']} json={rec['json_ok']} "
                    f"pred={rec['pred_label']} gt={gt_label} tools={rec['tool_calls']} t={rec['latency_ms']}ms"
                )
                tqdm.write(msg)
                lf.write(msg + "\n")
                lf.flush()

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
        "json_ok": int(json_ok_count),
        "json_fail": int(json_fail_count),
        "json_success_rate": json_ok_count / len(eval_df) if len(eval_df) else 0.0,
        "avg_latency_ms": total_latency / len(eval_df) if len(eval_df) else 0.0,
        "elapsed_seconds": round(time.time() - start_ts, 3),
        "avg_tool_calls": total_tool_calls / len(eval_df) if len(eval_df) else 0.0,
        "chat_endpoint": endpoint,
        "model_name": args.model_name,
        "tool_config": str(Path(args.tool_config).expanduser().resolve()),
        "tool_index": args.tool_index,
        "active_tools": [tool.name for tool in tools],
        "tool_call_counts": dict(tool_call_counter),
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

    with run_log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"[{datetime.now().isoformat(timespec='seconds')}] finished evaluation\n")
        lf.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    print("[DONE] vLLM baseline + tools evaluation finished")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baseline model with tool-calling on IFD_T eval.parquet")
    p.add_argument("--eval-input", default="/ssd2/jydai/data/IFD_T/ifd_t_105k_v1/eval.parquet")
    p.add_argument("--chat-endpoint", default="http://127.0.0.1:8010/v1/chat/completions")
    p.add_argument("--model-name", default="Qwen3vl8b")
    p.add_argument("--tool-config", default="/data/home/yxzhou/jydai/work/IFD_T/tools/verl_tool_config.yaml")
    p.add_argument("--tool-index", type=int, default=-1, help="Use one tool by index; -1 means all tools")
    p.add_argument("--max-tool-rounds", type=int, default=3)
    p.add_argument("--system-prompt", default="/data/home/yxzhou/jydai/work/IFD_T/prompt/system_prompt.md")
    p.add_argument("--user-template", default="/data/home/yxzhou/jydai/work/IFD_T/prompt/user_prompt_template.md")
    p.add_argument("--output-dir", default="/ssd2/jydai/data/IFD_T/ifd_t_105k_v1/eval_vllm_baseline_with_tools_qwen3vl8b")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-completion-tokens", type=int, default=4096)
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
