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
            name="srm_filter",
            description=(
                "Apply Spatial Rich Model high-order residual filter bank to suppress semantic "
                "content and expose sub-pixel manipulation noise traces."
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
                    "truncate_threshold": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Residual truncation threshold (default 3.0, range 1.0-10.0).",
                    ),
                    "export_map": OpenAIFunctionPropertySchema(
                        type="boolean",
                        description="Whether to export aggregated SRM heatmap image.",
                    ),
                    "output_dir": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Directory to save exported SRM heatmap.",
                    ),
                },
                required=["image_input"],
            ),
            strict=False,
        ),
    )


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        fvalue = float(value)
    except Exception:
        fvalue = default
    return max(low, min(high, fvalue))


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(arr))
    hi = float(np.percentile(arr, 99.0))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _conv2d_reflect(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(gray, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(gray, dtype=np.float32)

    for i in range(kh):
        for j in range(kw):
            coeff = float(kernel[i, j])
            if abs(coeff) < 1e-12:
                continue
            out += coeff * padded[i : i + gray.shape[0], j : j + gray.shape[1]]

    return out


def _build_srm_filter_bank() -> List[Tuple[str, np.ndarray, float]]:
    # Each tuple: (filter_name, kernel, normalization_scale)
    return [
        # First-order differential residuals
        ("fo_h", np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float32), 1.0),
        ("fo_v", np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32), 1.0),
        ("fo_d1", np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32), 1.0),
        ("fo_d2", np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], dtype=np.float32), 1.0),
        # Second-order linear residuals
        ("so_h", np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32), 2.0),
        ("so_v", np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32), 2.0),
        ("so_d1", np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.float32), 2.0),
        ("so_d2", np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float32), 2.0),
        # Third-order directional predictors
        (
            "to_h",
            np.array([[0, 0, 0, 0, 0], [0, 1, -3, 3, -1], [0, 0, 0, 0, 0]], dtype=np.float32),
            4.0,
        ),
        (
            "to_v",
            np.array([[0, 0, 0], [0, 1, 0], [0, -3, 0], [0, 3, 0], [0, -1, 0]], dtype=np.float32),
            4.0,
        ),
        # Fourth-order predictors
        (
            "fo4_h",
            np.array([[0, 0, 0, 0, 0], [1, -4, 6, -4, 1], [0, 0, 0, 0, 0]], dtype=np.float32),
            6.0,
        ),
        (
            "fo4_v",
            np.array([[0, 1, 0], [0, -4, 0], [0, 6, 0], [0, -4, 0], [0, 1, 0]], dtype=np.float32),
            6.0,
        ),
        # Nonlinear mixed predictors (cross-derivative-like)
        (
            "nl_mix1",
            np.array(
                [
                    [0, 0, -1, 0, 0],
                    [0, 2, 0, 2, 0],
                    [-1, 0, -4, 0, -1],
                    [0, 2, 0, 2, 0],
                    [0, 0, -1, 0, 0],
                ],
                dtype=np.float32,
            ),
            8.0,
        ),
        (
            "nl_mix2",
            np.array(
                [
                    [0, 1, 0, -1, 0],
                    [1, 0, -2, 0, 1],
                    [0, -2, 0, 2, 0],
                    [-1, 0, 2, 0, -1],
                    [0, 1, 0, -1, 0],
                ],
                dtype=np.float32,
            ),
            8.0,
        ),
        (
            "nl_mix3",
            np.array(
                [
                    [1, -2, 1],
                    [-2, 4, -2],
                    [1, -2, 1],
                ],
                dtype=np.float32,
            ),
            4.0,
        ),
        (
            "nl_mix4",
            np.array(
                [
                    [0, 1, -2, 1, 0],
                    [1, -4, 6, -4, 1],
                    [-2, 6, -8, 6, -2],
                    [1, -4, 6, -4, 1],
                    [0, 1, -2, 1, 0],
                ],
                dtype=np.float32,
            ),
            12.0,
        ),
    ]


class SRMFilter:
    def __init__(
        self,
        default_truncate_threshold: float = 3.0,
        default_export_map: bool = True,
        default_output_dir: str = "./outputs/srm_filter",
    ):
        self.tool_name = "SRM_Filter"
        self.default_truncate_threshold = default_truncate_threshold
        self.default_export_map = default_export_map
        self.default_output_dir = default_output_dir
        self.filter_bank = _build_srm_filter_bank()

    def _extract_luma(self, image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    def _residual_stats(self, residual: np.ndarray) -> Dict[str, float]:
        arr = residual.astype(np.float32).ravel()
        if arr.size == 0:
            return {"mean_abs": 0.0, "std": 0.0, "kurtosis": 0.0}

        mean_abs = float(np.mean(np.abs(arr)))
        std = float(np.std(arr))
        centered = arr - float(np.mean(arr))
        var = float(np.mean(centered ** 2))
        if var < 1e-8:
            kurtosis = 0.0
        else:
            kurtosis = float(np.mean(centered ** 4) / (var ** 2 + 1e-12) - 3.0)

        return {
            "mean_abs": round(mean_abs, 6),
            "std": round(std, 6),
            "kurtosis": round(kurtosis, 6),
        }

    def _aggregate_suspicion(self, residual_maps: List[np.ndarray]) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
        if not residual_maps:
            return np.zeros((1, 1), dtype=np.float32), 0.0, []

        stack = np.stack([np.abs(x).astype(np.float32) for x in residual_maps], axis=0)
        agg = np.sqrt(np.mean(stack ** 2, axis=0))

        q50 = float(np.percentile(agg, 50.0))
        q90 = float(np.percentile(agg, 90.0))
        q95 = float(np.percentile(agg, 95.0))
        q99 = float(np.percentile(agg, 99.0))

        strong_ratio = float(np.mean(agg >= q95))
        extreme_ratio = float(np.mean(agg >= q99))
        contrast_score = max(0.0, min(1.0, (q99 - q50) / (q99 + 1e-6)))
        tail_score = max(0.0, min(1.0, (q99 - q90) / (q99 + 1e-6)))

        tamper_prob = 0.45 * strong_ratio * 10.0 + 0.30 * extreme_ratio * 20.0 + 0.15 * contrast_score + 0.10 * tail_score
        tamper_prob = max(0.0, min(1.0, tamper_prob))

        h, w = agg.shape
        gh, gw = 3, 3
        top_regions: List[Dict[str, Any]] = []
        for gy in range(gh):
            for gx in range(gw):
                y0 = int(round(gy * h / gh))
                y1 = int(round((gy + 1) * h / gh))
                x0 = int(round(gx * w / gw))
                x1 = int(round((gx + 1) * w / gw))
                block = agg[y0:y1, x0:x1]
                score = float(np.mean(block)) if block.size > 0 else 0.0
                top_regions.append(
                    {
                        "region": f"x:{x0}-{max(x0, x1 - 1)}, y:{y0}-{max(y0, y1 - 1)}",
                        "score": score,
                    }
                )

        top_regions.sort(key=lambda x: x["score"], reverse=True)
        for item in top_regions[:3]:
            item["score"] = round(item["score"], 6)

        return agg, round(tamper_prob, 4), top_regions[:3]

    def _save_heatmap(self, agg_map: np.ndarray, image_path: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        u8 = _normalize_to_uint8(agg_map)
        heat = Image.fromarray(u8, mode="L").convert("RGB")

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_srm_heatmap.png")
        heat.save(out_path)
        return out_path

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

    def __call__(self, args_str: str) -> Dict[str, Any]:
        start_time = time.time()
        log_call_args: Dict[str, Any] = {"raw_args": args_str}

        def _finalize(response_obj: Dict[str, Any]) -> Dict[str, Any]:
            log_tool_use_record(
                tool_key="srm_filter",
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
                    description="SRM filter analysis failed.",
                    error_message="工具参数解析失败，必须是有效的 JSON 格式字符串。",
                )
            )

        image_path = args.get("image_path") or args.get("image_input")
        truncate_threshold = _clamp_float(
            args.get("truncate_threshold", self.default_truncate_threshold),
            low=1.0,
            high=10.0,
            default=self.default_truncate_threshold,
        )
        export_map = bool(args.get("export_map", self.default_export_map))
        output_dir = str(args.get("output_dir", self.default_output_dir))

        log_call_args = {
            "image_path": image_path,
            "truncate_threshold": truncate_threshold,
            "export_map": export_map,
            "output_dir": output_dir,
        }

        if not image_path:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="SRM filter analysis failed.",
                    error_message="缺少必填参数 'image_path'。",
                )
            )

        if not os.path.exists(image_path):
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="SRM filter analysis failed.",
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
                    description="SRM filter analysis failed.",
                    error_message=f"读取图像失败: {exc}",
                )
            )

        residual_details: List[Dict[str, Any]] = []
        residual_maps: List[np.ndarray] = []

        try:
            for name, kernel, scale in self.filter_bank:
                residual = _conv2d_reflect(gray, kernel) / float(scale)
                residual = np.clip(residual, -truncate_threshold, truncate_threshold)
                residual_maps.append(residual)
                stats = self._residual_stats(residual)
                residual_details.append(
                    {
                        "filter": name,
                        "kernel_shape": list(kernel.shape),
                        "mean_abs": stats["mean_abs"],
                        "std": stats["std"],
                        "kurtosis": stats["kurtosis"],
                    }
                )

            agg_map, _tamper_probability, top_regions = self._aggregate_suspicion(residual_maps)
            heatmap_path = ""
            if export_map:
                heatmap_path = self._save_heatmap(agg_map, image_path=image_path, output_dir=output_dir)

            avg_kurtosis = float(np.mean([x["kurtosis"] for x in residual_details])) if residual_details else 0.0

            result = {
                "forensics_report": {
                    "output_mode": "evidence_only",
                    "method": "Spatial Rich Model residual filter bank",
                    "filter_count": len(self.filter_bank),
                    "truncate_threshold": round(float(truncate_threshold), 4),
                    "top_suspicious_regions": top_regions,
                    "summary": "已输出 SRM 残差统计与可疑区域证据；不包含最终判定结论。",
                },
                "residual_statistics": residual_details,
                "evidence_scores": {
                    "avg_abs_kurtosis": round(float(abs(avg_kurtosis)), 6),
                },
                "artifacts": {
                    "srm_heatmap_path": heatmap_path,
                },
            }

            return _finalize(
                self._build_response(
                    success=True,
                    start_time=start_time,
                    description="SRM filter analysis completed.",
                    result=result,
                )
            )
        except Exception as exc:
            logger.exception("SRMFilter execution failed.")
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="SRM filter analysis failed.",
                    error_message=f"SRM 计算阶段发生错误: {exc}",
                )
            )


class SRMFilterTool(BaseTool):
    """verl-native wrapper aligned with BaseTool lifecycle."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema or _default_schema())
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self._default_truncate_threshold = float(config.get("default_truncate_threshold", 3.0))
        self._default_export_map = bool(config.get("default_export_map", True))
        self._default_output_dir = str(config.get("output_dir", "./outputs/srm_filter"))
        self._tool_impl = SRMFilter(
            default_truncate_threshold=self._default_truncate_threshold,
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
                "truncate_threshold": parameters.get(
                    "truncate_threshold",
                    self._default_truncate_threshold,
                ),
                "export_map": parameters.get("export_map", self._default_export_map),
                "output_dir": parameters.get("output_dir", self._default_output_dir),
            }
            result = self._tool_impl(json.dumps(call_args, ensure_ascii=False))
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["response"] = result
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as exc:
            logger.exception("SRMFilterTool execution failed.")
            error_result = {
                "success": False,
                "status": "error",
                "description": "SRM filter analysis failed.",
                "result": {},
                "error_message": f"Tool runtime error: {exc}",
            }
            return ToolResponse(text=json.dumps(error_result, ensure_ascii=False)), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
