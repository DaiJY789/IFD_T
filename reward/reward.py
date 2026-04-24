import json
import re
from typing import Any, Optional


ALLOWED_VERDICTS = {"tampered", "ai_manipulated", "authentic", "uncertain"}
TAMPER_TYPES = {"splicing", "copy_move", "inpainting", "retouching", "unknown"}
AI_TYPES = {"ai_eited", "ai_edited", "ai_generated"}

TOOL_NAME_MARKERS = (
    "srm_filter",
    "noise_analyzer",
    "patchdense_scanner",
    "cfa_validator",
    "spectrum_grid_analyzer",
    "facial_biological_auditor",
    "roi_extractor",
    "image_resizer",
    "visual_enhancer",
    "<tool_call>",
    "tool_calls",
)


def _to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    return value


def _as_float01(value: Any) -> tuple[Optional[float], bool]:
    try:
        v = float(value)
    except Exception:
        return None, False
    if 0.0 <= v <= 1.0:
        return v, True
    return v, False


def _normalize_verdict(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    if text in {"tampered", "tp", "tamper", "manipulated", "篡改", "伪造篡改", "拼接"}:
        return "tampered"
    if text in {
        "ai_manipulated",
        "ai_manipulate",
        "ai",
        "ag",
        "aigc",
        "ai_generated",
        "deepfake",
        "ai操纵",
        "ai伪造",
    }:
        return "ai_manipulated"
    if text in {"authentic", "re", "real", "genuine", "true", "真实", "原图", "真图"}:
        return "authentic"
    if text in {"uncertain", "unknown", "unsure", "不确定", "无法确定", "存疑"}:
        return "uncertain"

    if "ai" in text or "aigc" in text or "deepfake" in text:
        return "ai_manipulated"
    if "tamper" in text or "manip" in text or "篡改" in text:
        return "tampered"
    if "real" in text or "authentic" in text or "真实" in text:
        return "authentic"
    if "uncertain" in text or "不确定" in text:
        return "uncertain"

    return None


def _normalize_forgery_type(value: Any) -> Optional[str]:
    text = str(value).strip().lower()
    if not text:
        return None
    mapping = {
        "splice": "splicing",
        "splicing": "splicing",
        "copy-move": "copy_move",
        "copy_move": "copy_move",
        "copymove": "copy_move",
        "inpaint": "inpainting",
        "inpainting": "inpainting",
        "retouch": "retouching",
        "retouching": "retouching",
        "unknown": "unknown",
        "ai_generated": "ai_generated",
        "aigc": "ai_generated",
        "ai_gen": "ai_generated",
        "ai_edited": "ai_eited",
        "ai_eited": "ai_eited",
        "deepfake": "ai_generated",
    }
    return mapping.get(text)


def _extract_json_candidates(text: str) -> list[dict]:
    candidates: list[dict] = []

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            candidates.append(obj)
    except Exception:
        pass

    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
        block = match.group(1).strip()
        if not block:
            continue
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                candidates.append(obj)
                continue
        except Exception:
            pass

        inner = re.search(r"\{[\s\S]*\}", block)
        if not inner:
            continue
        try:
            obj = json.loads(inner.group(0))
            if isinstance(obj, dict):
                candidates.append(obj)
        except Exception:
            pass

    decoder = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                candidates.append(obj)
            i += max(end, 1)
        except Exception:
            i += 1

    return candidates


def _pick_best_candidate(candidates: list[dict]) -> Optional[dict]:
    if not candidates:
        return None

    required = {
        "verdict",
        "probability",
        "confidence",
        "forgery_types",
        "evidence_chain",
        "suspicious_regions",
        "consistency_check",
        "limitations",
        "final_summary",
    }

    hint_keys = required | {"label", "prediction", "pred_label"}

    best_obj: Optional[dict] = None
    best_score = -1
    for obj in candidates:
        keys = set(obj.keys())
        if not (keys & hint_keys):
            continue
        score = len(required & keys)
        if "verdict" in keys:
            score += 3
        if score > best_score:
            best_score = score
            best_obj = obj
    return best_obj


def _infer_verdict_from_text(solution_str: str) -> Optional[str]:
    text = (solution_str or "")
    if not text:
        return None

    patterns = [
        r'"verdict"\s*:\s*"?([a-zA-Z_]+)"?',
        r'"label"\s*:\s*"?([a-zA-Z_]+)"?',
        r'\bverdict\s*[:=]\s*"?([a-zA-Z_]+)"?',
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        normalized = _normalize_verdict(m.group(1))
        if normalized is not None:
            return normalized
    return None


def _parse_response(solution_str: str) -> tuple[Optional[dict], bool]:
    text = (solution_str or "").strip()
    if not text:
        return None, False

    candidates = _extract_json_candidates(text)
    parsed = _pick_best_candidate(candidates)
    if isinstance(parsed, dict):
        return parsed, True
    return None, False


def _parse_gt(ground_truth: Any, extra_info: Optional[dict]) -> Optional[str]:
    gt = _normalize_verdict(ground_truth)
    if gt is not None:
        return gt
    if isinstance(extra_info, dict):
        for key in ("label", "gt", "ground_truth", "answer", "target", "verdict"):
            gt = _normalize_verdict(extra_info.get(key))
            if gt is not None:
                return gt
    return None


def _parse_types(values: Any) -> set[str]:
    if values is None:
        return set()
    if not isinstance(values, list):
        values = [values]

    out = set()
    for x in values:
        t = _normalize_forgery_type(x)
        if t is not None:
            out.add(t)
    return out


def _extract_gt_types(ground_truth: Any, extra_info: Optional[dict]) -> set[str]:
    if isinstance(ground_truth, dict):
        for key in ("forgery_types", "forgery_type", "types", "type"):
            if key in ground_truth:
                parsed = _parse_types(ground_truth.get(key))
                if parsed:
                    return parsed

    if isinstance(extra_info, dict):
        for key in ("forgery_types", "forgery_type", "target_forgery_types", "gt_forgery_types"):
            if key in extra_info:
                parsed = _parse_types(extra_info.get(key))
                if parsed:
                    return parsed

    return set()


def _extract_pred_types(parsed: Optional[dict]) -> set[str]:
    if not isinstance(parsed, dict):
        return set()
    return _parse_types(parsed.get("forgery_types"))


def _estimate_tool_usage(solution_str: str, parsed: Optional[dict], extra_info: Optional[dict]) -> int:
    calls = 0
    if isinstance(extra_info, dict):
        turns = extra_info.get("num_turns")
        if isinstance(turns, int) and turns > 1:
            calls = max(calls, turns - 1)

    if isinstance(parsed, dict):
        chain = parsed.get("evidence_chain")
        if isinstance(chain, list):
            calls = max(calls, sum(1 for x in chain if isinstance(x, dict) and str(x.get("tool", "")).strip()))

    text = (solution_str or "").lower()
    calls = max(calls, sum(1 for marker in TOOL_NAME_MARKERS if marker in text))
    return calls


def _minimal_schema_ok(parsed: Optional[dict]) -> tuple[bool, Optional[str], bool, bool]:
    if not isinstance(parsed, dict):
        return False, None, False, False

    required = {
        "verdict",
        "probability",
        "confidence",
        "forgery_types",
        "evidence_chain",
        "suspicious_regions",
        "consistency_check",
        "limitations",
        "final_summary",
    }
    has_required = required.issubset(set(parsed.keys()))
    verdict = _normalize_verdict(parsed.get("verdict"))
    _, p_ok = _as_float01(parsed.get("probability"))
    _, c_ok = _as_float01(parsed.get("confidence"))
    return has_required and p_ok and c_ok and verdict in ALLOWED_VERDICTS, verdict, p_ok, c_ok


def _semantic_consistency_ok(parsed: Optional[dict], verdict: Optional[str]) -> bool:
    if not isinstance(parsed, dict) or verdict is None:
        return False

    ftypes_raw = parsed.get("forgery_types")
    regions = parsed.get("suspicious_regions")
    if not isinstance(ftypes_raw, list) or not isinstance(regions, list):
        return False

    ftypes = {_normalize_forgery_type(x) for x in ftypes_raw}
    ftypes.discard(None)

    if verdict == "tampered":
        return bool(ftypes) and ftypes.issubset(TAMPER_TYPES) and len(regions) >= 1
    if verdict == "ai_manipulated":
        if not ftypes.issubset(AI_TYPES) or not ftypes:
            return False
        if ftypes == {"ai_generated"}:
            return len(regions) == 0
        return len(regions) >= 1
    if verdict == "authentic":
        return len(ftypes) == 0 and len(regions) == 0
    if verdict == "uncertain":
        return ftypes in (set(), {"unknown"}) and len(regions) == 0
    return False


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Simplified reward with <=5 core components.

    1) classification correctness
    2) JSON format/schema validity
    3) verdict-semantic consistency
    4) tool usage encouragement
    5) optional subtype alignment (forgery_types) when GT is available
    """
    parsed, json_ok = _parse_response(solution_str)
    schema_ok, pred, p_ok, c_ok = _minimal_schema_ok(parsed)
    if pred is None:
        pred = _infer_verdict_from_text(solution_str)
    semantic_ok = _semantic_consistency_ok(parsed, pred)
    gt = _parse_gt(ground_truth, extra_info if isinstance(extra_info, dict) else None)
    gt_types = _extract_gt_types(ground_truth, extra_info if isinstance(extra_info, dict) else None)
    pred_types = _extract_pred_types(parsed)
    tool_calls = _estimate_tool_usage(
        solution_str,
        parsed if isinstance(parsed, dict) else None,
        extra_info if isinstance(extra_info, dict) else None,
    )

    # Core 1: correctness
    if pred is None:
        score_correct = -0.6
        acc = False
    elif gt is None:
        score_correct = 0.0
        acc = False
    elif pred == gt:
        score_correct = 1.0
        acc = True
    else:
        score_correct = -0.7
        acc = False

    # Core 2: format/schema
    score_format = 0.25 if (json_ok and schema_ok and p_ok and c_ok) else -0.30

    # Core 3: semantic consistency
    score_semantic = 0.20 if semantic_ok else -0.20

    # Core 4: tool usage
    if tool_calls == 0:
        score_tool = -0.1
    elif tool_calls == 1:
        score_tool = 0.08
    else:
        score_tool = 0.15

    # Core 5: subtype reward only for samples with subtype GT.
    # This avoids punishing tampered samples that do not carry forgery_types labels.
    if gt_types and gt == "ai_manipulated":
        inter = len(gt_types & pred_types)
        union = len(gt_types | pred_types)
        jaccard = (inter / union) if union > 0 else 0.0
        score_type2 = 0.20 * jaccard - 0.10  # [-0.10, +0.10]
        type2_active = True
    else:
        score_type2 = 0.0
        type2_active = False

    score = score_correct + score_format + score_semantic + score_tool + score_type2
    score = max(-1.5, min(1.5, score))

    result = {
        "score": float(score),
        "acc": bool(acc),
        "pred": pred or "",
        "gt": gt or "",
        "json_ok": bool(json_ok),
        "schema_ok": bool(schema_ok),
        "semantic_ok": bool(semantic_ok),
        "type2_active": bool(type2_active),
        # Keep type sets as strings to avoid list aggregation failures in validation metrics.
        "gt_types": "|".join(sorted(gt_types)),
        "pred_types": "|".join(sorted(pred_types)),
        "gt_types_count": int(len(gt_types)),
        "pred_types_count": int(len(pred_types)),
        "score_type2": float(score_type2),
        "tool_calls": int(tool_calls),
        "data_source": str(data_source),
    }
    return _to_native(result)
