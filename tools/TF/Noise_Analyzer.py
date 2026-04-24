import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from PIL import Image
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)
from verl.utils.rollout_trace import rollout_trace_op

try:
    from work.tool_llm.tool_use_logger import log_tool_use_record
except Exception:

    def log_tool_use_record(**kwargs):  # type: ignore
        return None


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _default_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="noise_analyzer",
            description=(
                "Detect splicing risk via wavelet-domain noise variance inconsistency. "
                "Outputs local noise variance heatmap and suspicious regions."
            ),
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "image_input": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Local image path to analyze.",
                    ),
                    "image_path": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Alias of image_input.",
                    ),
                    "window_size": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Local variance window size (odd number, default 9, range 5-31).",
                    ),
                    "levels": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Haar wavelet decomposition levels (default 2, range 1-3).",
                    ),
                    "export_map": OpenAIFunctionPropertySchema(
                        type="boolean",
                        description="Whether to export noise variance heatmap image.",
                    ),
                    "output_dir": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Directory to save exported heatmap image.",
                    ),
                },
                required=["image_input"],
            ),
            strict=False,
        ),
    )


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        ivalue = int(round(float(value)))
    except Exception:
        ivalue = default
    return max(low, min(high, ivalue))


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(arr))
    hi = float(np.percentile(arr, 99.0))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _box_mean(arr: np.ndarray, win: int) -> np.ndarray:
    pad = win // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
    integ = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
    h, w = arr.shape
    y0 = np.arange(0, h)
    y1 = y0 + win
    x0 = np.arange(0, w)
    x1 = x0 + win
    out = (
        integ[np.ix_(y1, x1)]
        - integ[np.ix_(y0, x1)]
        - integ[np.ix_(y1, x0)]
        + integ[np.ix_(y0, x0)]
    )
    return out / float(win * win)


def _local_variance(arr: np.ndarray, win: int) -> np.ndarray:
    mean = _box_mean(arr, win)
    mean2 = _box_mean(arr * arr, win)
    var = np.maximum(mean2 - mean * mean, 0.0)
    return var.astype(np.float32)


def _haar_dwt2_once(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = gray.shape
    if h % 2 == 1:
        gray = gray[:-1, :]
    if w % 2 == 1:
        gray = gray[:, :-1]

    a = gray[0::2, 0::2]
    b = gray[0::2, 1::2]
    c = gray[1::2, 0::2]
    d = gray[1::2, 1::2]

    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll.astype(np.float32), lh.astype(np.float32), hl.astype(np.float32), hh.astype(np.float32)


def _upsample_repeat(arr: np.ndarray, target_h: int, target_w: int, factor: int) -> np.ndarray:
    up = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
    if up.shape[0] < target_h or up.shape[1] < target_w:
        pad_h = max(0, target_h - up.shape[0])
        pad_w = max(0, target_w - up.shape[1])
        up = np.pad(up, ((0, pad_h), (0, pad_w)), mode="edge")
    return up[:target_h, :target_w]


class NoiseAnalyzer:
    def __init__(
        self,
        default_window_size: int = 9,
        default_levels: int = 2,
        default_export_map: bool = True,
        default_output_dir: str = "./outputs/noise_analyzer",
    ):
        self.tool_name = "Noise_Analyzer"
        self.default_window_size = default_window_size
        self.default_levels = default_levels
        self.default_export_map = default_export_map
        self.default_output_dir = default_output_dir

    def _extract_luma(self, image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    def _build_response(
        self,
        success: bool,
        start_time: float,
        description: str,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        execution_time_ms = int((time.time() - start_time) * 1000)
        return {
            "success": success,
            "status": "success" if success else "error",
            "description": description,
            "result": result or {},
            "error_message": error_message,
            "metadata": {
                "tool_name": self.tool_name,
                "execution_time_ms": execution_time_ms,
            },
        }

    def _estimate_noise_var_map(
        self, gray: np.ndarray, levels: int, window_size: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        h0, w0 = gray.shape
        ll = gray
        hh_var_upsampled: List[np.ndarray] = []
        level_stats: List[Dict[str, float]] = []

        for lv in range(levels):
            ll, _, _, hh = _haar_dwt2_once(ll)
            var_hh = _local_variance(hh, window_size)
            factor = 2 ** (lv + 1)
            var_up = _upsample_repeat(var_hh, h0, w0, factor=factor)
            hh_var_upsampled.append(var_up)

            level_stats.append(
                {
                    "level": float(lv + 1),
                    "hh_var_mean": float(np.mean(var_hh)),
                    "hh_var_std": float(np.std(var_hh)),
                }
            )

        if not hh_var_upsampled:
            var_map = np.zeros((h0, w0), dtype=np.float32)
        else:
            var_map = np.mean(np.stack(hh_var_upsampled, axis=0), axis=0).astype(np.float32)

        mu = float(np.mean(var_map))
        sigma = float(np.std(var_map)) + 1e-8
        z = (var_map - mu) / sigma

        strong_ratio = float(np.mean(z > 2.0))
        extreme_ratio = float(np.mean(z > 3.0))
        tamper_prob = 0.55 * strong_ratio * 7.5 + 0.45 * extreme_ratio * 13.0
        tamper_prob = max(0.0, min(1.0, tamper_prob))

        detail = {
            "global_var_mean": mu,
            "global_var_std": sigma,
            "strong_ratio": strong_ratio,
            "extreme_ratio": extreme_ratio,
            "level_stats": level_stats,
        }
        return var_map, {"tamper_probability": tamper_prob, "detail": detail}

    def _top_regions(self, score_map: np.ndarray, topk: int = 3) -> List[Dict[str, Any]]:
        h, w = score_map.shape
        gh, gw = 3, 3
        regions: List[Dict[str, Any]] = []

        for gy in range(gh):
            for gx in range(gw):
                y0 = int(round(gy * h / gh))
                y1 = int(round((gy + 1) * h / gh))
                x0 = int(round(gx * w / gw))
                x1 = int(round((gx + 1) * w / gw))
                block = score_map[y0:y1, x0:x1]
                score = float(np.mean(block)) if block.size > 0 else 0.0
                regions.append(
                    {
                        "region": f"x:{x0}-{max(x0, x1 - 1)}, y:{y0}-{max(y0, y1 - 1)}",
                        "score": round(score, 6),
                    }
                )

        regions.sort(key=lambda x: x["score"], reverse=True)
        return regions[:topk]

    def _save_heatmap(self, var_map: np.ndarray, image_path: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        heat_u8 = _normalize_to_uint8(var_map)
        heat_img = Image.fromarray(heat_u8, mode="L").convert("RGB")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_noise_var_heatmap.png")
        heat_img.save(out_path)
        return out_path

    def __call__(self, args_str: str) -> Dict[str, Any]:
        start_time = time.time()
        log_call_args: Dict[str, Any] = {"raw_args": args_str}

        def _finalize(response_obj: Dict[str, Any]) -> Dict[str, Any]:
            log_tool_use_record(
                tool_key="noise_analyzer",
                tool_name=self.tool_name,
                call_args=log_call_args,
                response=response_obj,
            )
            return response_obj

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Noise variance analysis failed.",
                    error_message="工具参数解析失败，必须是有效的 JSON 格式字符串。",
                )
            )

        image_path = args.get("image_path") or args.get("image_input")
        window_size = _clamp_int(
            args.get("window_size", self.default_window_size),
            low=5,
            high=31,
            default=self.default_window_size,
        )
        if window_size % 2 == 0:
            window_size += 1

        levels = _clamp_int(
            args.get("levels", self.default_levels),
            low=1,
            high=3,
            default=self.default_levels,
        )
        export_map = bool(args.get("export_map", self.default_export_map))
        output_dir = str(args.get("output_dir", self.default_output_dir))

        log_call_args = {
            "image_path": image_path,
            "window_size": window_size,
            "levels": levels,
            "export_map": export_map,
            "output_dir": output_dir,
        }

        if not image_path:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Noise variance analysis failed.",
                    error_message="缺少必填参数 'image_path'。",
                )
            )

        if not os.path.exists(image_path):
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Noise variance analysis failed.",
                    error_message=f"找不到指定的图像文件: {image_path}。请检查文件路径是否正确。",
                )
            )

        try:
            image = Image.open(image_path).convert("RGB")
            gray = self._extract_luma(image)
        except Exception as exc:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Noise variance analysis failed.",
                    error_message=f"读取图像失败: {exc}",
                )
            )

        try:
            var_map, score_info = self._estimate_noise_var_map(
                gray=gray,
                levels=levels,
                window_size=window_size,
            )

            tamper_probability = float(score_info["tamper_probability"])
            top_regions = self._top_regions(var_map, topk=3)

            global_std = float(score_info["detail"]["global_var_std"])
            evidence_score = max(0.0, min(1.0, tamper_probability))

            heatmap_path = ""
            if export_map:
                heatmap_path = self._save_heatmap(var_map, image_path=image_path, output_dir=output_dir)

            result = {
                "forensics_report": {
                    "output_mode": "evidence_only",
                    "method": "Wavelet HH-band + Local Noise Variance Inconsistency",
                    "levels": levels,
                    "window_size": window_size,
                    "top_suspicious_regions": top_regions,
                    "summary": "已输出噪声方差不一致统计与局部区域证据；不包含最终判定结论。",
                },
                "evidence_scores": {
                    "noise_inconsistency_score": round(float(evidence_score), 6),
                },
                "noise_statistics": {
                    "global_var_mean": round(float(score_info["detail"]["global_var_mean"]), 6),
                    "global_var_std": round(float(score_info["detail"]["global_var_std"]), 6),
                    "strong_ratio": round(float(score_info["detail"]["strong_ratio"]), 6),
                    "extreme_ratio": round(float(score_info["detail"]["extreme_ratio"]), 6),
                    "level_stats": [
                        {
                            "level": int(x["level"]),
                            "hh_var_mean": round(float(x["hh_var_mean"]), 6),
                            "hh_var_std": round(float(x["hh_var_std"]), 6),
                        }
                        for x in score_info["detail"]["level_stats"]
                    ],
                },
                "artifacts": {
                    "noise_variance_heatmap_path": heatmap_path,
                },
            }

            return _finalize(
                self._build_response(
                    success=True,
                    start_time=start_time,
                    description="Noise variance inconsistency analysis completed.",
                    result=result,
                )
            )
        except Exception as exc:
            logger.exception("NoiseAnalyzer execution failed.")
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Noise variance analysis failed.",
                    error_message=f"噪声方差计算阶段发生错误: {exc}",
                )
            )


class NoiseAnalyzerTool(BaseTool):
    """verl-native wrapper aligned with BaseTool lifecycle."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema or _default_schema())
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self._default_window_size = int(config.get("default_window_size", 9))
        self._default_levels = int(config.get("default_levels", 2))
        self._default_export_map = bool(config.get("default_export_map", True))
        self._default_output_dir = str(config.get("output_dir", "./outputs/noise_analyzer"))
        self._tool_impl = NoiseAnalyzer(
            default_window_size=self._default_window_size,
            default_levels=self._default_levels,
            default_export_map=self._default_export_map,
            default_output_dir=self._default_output_dir,
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        default_image_path = kwargs.get("image_input") or kwargs.get("image_path")
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
            "default_image_path": default_image_path,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        env_states: Optional[dict] = None,
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        try:
            default_image_path = self._instance_dict.get(instance_id, {}).get("default_image_path")
            call_args = {
                "image_path": (
                    kwargs.get("image_input")
                    or kwargs.get("image_path")
                    or parameters.get("image_input")
                    or parameters.get("image_path")
                    or default_image_path
                ),
                "window_size": parameters.get("window_size", self._default_window_size),
                "levels": parameters.get("levels", self._default_levels),
                "export_map": parameters.get("export_map", self._default_export_map),
                "output_dir": parameters.get("output_dir", self._default_output_dir),
            }
            result = self._tool_impl(json.dumps(call_args, ensure_ascii=False))
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["response"] = result
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as exc:
            logger.exception("NoiseAnalyzerTool execution failed.")
            error_result = {
                "success": False,
                "status": "error",
                "description": "Noise variance analysis failed.",
                "result": {},
                "error_message": f"Tool runtime error: {exc}",
            }
            return ToolResponse(text=json.dumps(error_result, ensure_ascii=False)), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
