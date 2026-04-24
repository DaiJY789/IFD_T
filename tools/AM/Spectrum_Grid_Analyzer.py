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
            name="spectrum_grid_analyzer",
            description=(
                "Analyze deepfake/AI-generation artifacts via 2D DFT spectrum. Detect off-axis grid peaks, "
                "energy-hole anomalies, and map anomalies back to spatial domain via inverse transform."
            ),
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "image_input": OpenAIFunctionPropertySchema(type="string", description="Local image path."),
                    "image_path": OpenAIFunctionPropertySchema(type="string", description="Alias of image_input."),
                    "peak_sigma": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Peak threshold in sigma over background on log spectrum, range 1.5-6.0.",
                    ),
                    "center_exclusion_ratio": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Center exclusion radius ratio for off-axis peak detection, range 0.03-0.25.",
                    ),
                    "energy_hole_sensitivity": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Sensitivity for radial energy abrupt-drop detection, range 1.0-5.0.",
                    ),
                    "max_side": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Max image side for processing speed, range 256-1024.",
                    ),
                    "export_map": OpenAIFunctionPropertySchema(
                        type="boolean",
                        description="Whether to export frequency and spatial artifact maps.",
                    ),
                    "output_dir": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Directory to save generated maps.",
                    ),
                },
                required=["image_input"],
            ),
            strict=False,
        ),
    )


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default


def _clip(v: float, low: float, high: float) -> float:
    return max(low, min(high, v))


def _normalize_u8(arr: np.ndarray, p_hi: float = 99.0) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(arr))
    hi = float(np.percentile(arr, p_hi))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _radial_profile(energy: np.ndarray, cx: int, cy: int) -> np.ndarray:
    h, w = energy.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ridx = np.floor(rr).astype(np.int32)
    max_r = int(np.max(ridx))

    radial_sum = np.bincount(ridx.ravel(), weights=energy.ravel(), minlength=max_r + 1)
    radial_cnt = np.bincount(ridx.ravel(), minlength=max_r + 1)
    profile = radial_sum / np.maximum(radial_cnt, 1)
    return profile.astype(np.float32)


def _moving_average_1d(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr.copy()
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


class SpectrumGridAnalyzer:
    def __init__(
        self,
        default_peak_sigma: float = 3.0,
        default_center_exclusion_ratio: float = 0.08,
        default_energy_hole_sensitivity: float = 2.2,
        default_max_side: int = 768,
        default_export_map: bool = True,
        default_output_dir: str = "./outputs/spectrum_grid_analyzer",
    ):
        self.tool_name = "Spectrum_Grid_Analyzer"
        self.default_peak_sigma = default_peak_sigma
        self.default_center_exclusion_ratio = default_center_exclusion_ratio
        self.default_energy_hole_sensitivity = default_energy_hole_sensitivity
        self.default_max_side = default_max_side
        self.default_export_map = default_export_map
        self.default_output_dir = default_output_dir

    def _prepare_gray(self, image: Image.Image, max_side: int) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        ow, oh = image.size
        scale = 1.0
        if max(ow, oh) > max_side:
            scale = float(max_side) / float(max(ow, oh))
        nw = max(64, int(round(ow * scale)))
        nh = max(64, int(round(oh * scale)))
        resized = image.resize((nw, nh), Image.BILINEAR).convert("L")
        gray = np.asarray(resized, dtype=np.float32)
        return gray, (ow, oh), (nw, nh)

    def _detect_grid_peaks(
        self,
        log_mag: np.ndarray,
        center_exclusion_ratio: float,
        peak_sigma: float,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, float]]:
        h, w = log_mag.shape
        cx, cy = w // 2, h // 2
        yy, xx = np.indices((h, w))
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r0 = max(4.0, min(h, w) * center_exclusion_ratio)

        valid = rr >= r0
        bg = log_mag[valid]
        mu = float(np.mean(bg)) if bg.size else float(np.mean(log_mag))
        std = float(np.std(bg)) + 1e-8
        thr = mu + peak_sigma * std

        candidate = (log_mag >= thr) & valid

        # Local maximum constraint (3x3 neighborhood)
        padded = np.pad(log_mag, ((1, 1), (1, 1)), mode="edge")
        local_max = np.ones_like(log_mag, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                nei = padded[dy : dy + h, dx : dx + w]
                local_max &= log_mag >= nei

        peak_mask = candidate & local_max
        ys, xs = np.where(peak_mask)

        peaks: List[Dict[str, Any]] = []
        for y, x in zip(ys, xs):
            peaks.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "dx_from_center": int(x - cx),
                    "dy_from_center": int(y - cy),
                    "strength": round(float(log_mag[y, x]), 6),
                    "radius": round(float(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)), 4),
                }
            )

        peaks.sort(key=lambda p: p["strength"], reverse=True)
        raw_peak_count = len(peaks)
        peaks = peaks[:80]

        peak_ratio = float(np.mean(peak_mask))
        stats = {
            "threshold": float(thr),
            "background_mean": float(mu),
            "background_std": float(std),
            "peak_count": float(len(peaks)),
            "raw_peak_count": float(raw_peak_count),
            "peak_ratio": float(peak_ratio),
        }
        return peak_mask.astype(np.uint8), peaks, stats

    def _detect_energy_hole(
        self,
        mag: np.ndarray,
        sensitivity: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = mag.shape
        cx, cy = w // 2, h // 2

        power = (mag ** 2).astype(np.float32)
        profile = _radial_profile(power, cx=cx, cy=cy)
        profile = profile / (float(np.max(profile)) + 1e-8)

        smooth = _moving_average_1d(profile, k=7)
        d1 = np.diff(smooth)

        mu = float(np.mean(d1))
        std = float(np.std(d1)) + 1e-8
        hole_thr = mu - sensitivity * std
        hole_idx = np.where(d1 < hole_thr)[0]

        hole_mask = np.zeros((h, w), dtype=np.uint8)
        yy, xx = np.indices((h, w))
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        for idx in hole_idx:
            r_lo = max(0.0, float(idx) - 0.5)
            r_hi = float(idx) + 1.5
            band = (rr >= r_lo) & (rr < r_hi)
            hole_mask[band] = 1

        hole_stats = {
            "hole_count": int(len(hole_idx)),
            "hole_indices": [int(i) for i in hole_idx[:32]],
            "hole_threshold": round(float(hole_thr), 8),
            "decay_mean": round(float(mu), 8),
            "decay_std": round(float(std), 8),
        }
        return hole_mask, hole_stats

    def _inverse_map_artifact(
        self,
        fft_shift: np.ndarray,
        peak_mask: np.ndarray,
        hole_mask: np.ndarray,
    ) -> np.ndarray:
        anomaly_mask = np.clip(peak_mask.astype(np.float32) + hole_mask.astype(np.float32), 0.0, 1.0)
        anomaly_fft = fft_shift * anomaly_mask
        inv = np.fft.ifft2(np.fft.ifftshift(anomaly_fft))
        spatial = np.abs(inv).astype(np.float32)
        return spatial

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
                tool_key="spectrum_grid_analyzer",
                tool_name=self.tool_name,
                call_args=log_call_args,
                response=response_obj,
            )
            return response_obj

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            return _finalize(
                self._build_response(False, start_time, "Spectrum analysis failed.", error_message="工具参数解析失败，必须是有效 JSON 字符串。")
            )

        image_path = args.get("image_path") or args.get("image_input")
        if not image_path:
            return _finalize(
                self._build_response(False, start_time, "Spectrum analysis failed.", error_message="缺少必填参数 image_path。")
            )
        if not os.path.exists(image_path):
            return _finalize(
                self._build_response(False, start_time, "Spectrum analysis failed.", error_message=f"找不到图像文件: {image_path}")
            )

        peak_sigma = _clip(_safe_float(args.get("peak_sigma", self.default_peak_sigma), self.default_peak_sigma), 1.5, 6.0)
        center_exclusion_ratio = _clip(
            _safe_float(args.get("center_exclusion_ratio", self.default_center_exclusion_ratio), self.default_center_exclusion_ratio),
            0.03,
            0.25,
        )
        energy_hole_sensitivity = _clip(
            _safe_float(args.get("energy_hole_sensitivity", self.default_energy_hole_sensitivity), self.default_energy_hole_sensitivity),
            1.0,
            5.0,
        )
        max_side = max(256, min(1024, _safe_int(args.get("max_side", self.default_max_side), self.default_max_side)))
        export_map = bool(args.get("export_map", self.default_export_map))
        output_dir = str(args.get("output_dir", self.default_output_dir))

        log_call_args = {
            "image_path": image_path,
            "peak_sigma": peak_sigma,
            "center_exclusion_ratio": center_exclusion_ratio,
            "energy_hole_sensitivity": energy_hole_sensitivity,
            "max_side": max_side,
            "export_map": export_map,
            "output_dir": output_dir,
        }

        try:
            image = Image.open(image_path).convert("RGB")
            gray, orig_size, proc_size = self._prepare_gray(image, max_side=max_side)

            fft2 = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft2)
            mag = np.abs(fft_shift).astype(np.float32)
            log_mag = np.log1p(mag)

            peak_mask, peaks, peak_stats = self._detect_grid_peaks(
                log_mag=log_mag,
                center_exclusion_ratio=center_exclusion_ratio,
                peak_sigma=peak_sigma,
            )

            hole_mask, hole_stats = self._detect_energy_hole(
                mag=mag,
                sensitivity=energy_hole_sensitivity,
            )

            spatial_map = self._inverse_map_artifact(
                fft_shift=fft_shift,
                peak_mask=peak_mask,
                hole_mask=hole_mask,
            )

            peak_ratio = float(peak_stats["peak_ratio"])
            raw_peak_count = int(peak_stats.get("raw_peak_count", peak_stats["peak_count"]))
            hole_count = int(hole_stats["hole_count"])
            spatial_mean = float(np.mean(spatial_map)) + 1e-8
            spatial_p99 = float(np.percentile(spatial_map, 99.0))
            spatial_concentration = min(1.0, spatial_p99 / (6.0 * spatial_mean))
            peak_density_score = min(1.0, raw_peak_count / 180.0)
            peak_sparsity_score = 1.0 - min(1.0, peak_ratio / 0.00022)
            hole_score = min(1.0, hole_count / 10.0)

            evidence_scores = {
                "peak_sparsity_score": round(float(peak_sparsity_score), 6),
                "peak_density_score": round(float(peak_density_score), 6),
                "energy_hole_score": round(float(hole_score), 6),
                "spatial_concentration_score": round(float(spatial_concentration), 6),
            }

            spectrum_path = ""
            peak_mask_path = ""
            spatial_map_path = ""
            if export_map:
                os.makedirs(output_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(image_path))[0]

                spectrum_u8 = _normalize_u8(log_mag)
                peak_u8 = (peak_mask * 255).astype(np.uint8)
                spatial_u8 = _normalize_u8(spatial_map)

                spectrum_path = os.path.join(output_dir, f"{stem}_spectrum_logmag.png")
                peak_mask_path = os.path.join(output_dir, f"{stem}_grid_peak_mask.png")
                spatial_map_path = os.path.join(output_dir, f"{stem}_artifact_spatial_map.png")

                Image.fromarray(spectrum_u8, mode="L").save(spectrum_path)
                Image.fromarray(peak_u8, mode="L").save(peak_mask_path)
                Image.fromarray(spatial_u8, mode="L").save(spatial_map_path)

            result = {
                "forensics_report": {
                    "output_mode": "evidence_only",
                    "method": "2D DFT + Grid Peaks + Energy Hole + Inverse Mapping",
                    "summary": "已输出频域峰值、能量空洞与空间映射证据；不包含最终判定结论。",
                },
                "spectrum_statistics": {
                    "peak_count": int(peak_stats["peak_count"]),
                    "raw_peak_count": int(raw_peak_count),
                    "peak_ratio": round(float(peak_ratio), 8),
                    "peak_threshold": round(float(peak_stats["threshold"]), 8),
                    "energy_hole_count": int(hole_count),
                    "energy_hole_indices": hole_stats.get("hole_indices", []),
                    "processed_size": {"width": int(proc_size[0]), "height": int(proc_size[1])},
                    "original_size": {"width": int(orig_size[0]), "height": int(orig_size[1])},
                },
                "evidence_scores": evidence_scores,
                "top_grid_peaks": peaks[:20],
                "artifacts": {
                    "spectrum_logmag_path": spectrum_path,
                    "grid_peak_mask_path": peak_mask_path,
                    "artifact_spatial_map_path": spatial_map_path,
                },
            }

            return _finalize(self._build_response(True, start_time, "Spectrum grid analysis completed.", result=result))
        except Exception as exc:
            logger.exception("SpectrumGridAnalyzer execution failed.")
            return _finalize(
                self._build_response(
                    False,
                    start_time,
                    "Spectrum analysis failed.",
                    error_message=f"运行时错误: {exc}",
                )
            )


class SpectrumGridAnalyzerTool(BaseTool):
    """verl-native wrapper aligned with BaseTool lifecycle."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema or _default_schema())
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self._default_peak_sigma = float(config.get("default_peak_sigma", 3.0))
        self._default_center_exclusion_ratio = float(config.get("default_center_exclusion_ratio", 0.08))
        self._default_energy_hole_sensitivity = float(config.get("default_energy_hole_sensitivity", 2.2))
        self._default_max_side = int(config.get("default_max_side", 768))
        self._default_export_map = bool(config.get("default_export_map", True))
        self._default_output_dir = str(config.get("output_dir", "./outputs/spectrum_grid_analyzer"))

        self._tool_impl = SpectrumGridAnalyzer(
            default_peak_sigma=self._default_peak_sigma,
            default_center_exclusion_ratio=self._default_center_exclusion_ratio,
            default_energy_hole_sensitivity=self._default_energy_hole_sensitivity,
            default_max_side=self._default_max_side,
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
            call_args = dict(parameters)
            call_args["image_path"] = (
                kwargs.get("image_input")
                or kwargs.get("image_path")
                or parameters.get("image_input")
                or parameters.get("image_path")
                or default_image_path
            )
            call_args.setdefault("peak_sigma", self._default_peak_sigma)
            call_args.setdefault("center_exclusion_ratio", self._default_center_exclusion_ratio)
            call_args.setdefault("energy_hole_sensitivity", self._default_energy_hole_sensitivity)
            call_args.setdefault("max_side", self._default_max_side)
            call_args.setdefault("export_map", self._default_export_map)
            call_args.setdefault("output_dir", self._default_output_dir)

            result = self._tool_impl(json.dumps(call_args, ensure_ascii=False))
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["response"] = result
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as exc:
            logger.exception("SpectrumGridAnalyzerTool execution failed.")
            error_result = {
                "success": False,
                "status": "error",
                "description": "Spectrum analysis failed.",
                "result": {},
                "error_message": f"Tool runtime error: {exc}",
            }
            return ToolResponse(text=json.dumps(error_result, ensure_ascii=False)), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
