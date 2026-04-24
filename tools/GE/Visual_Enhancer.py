import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple
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
            name="visual_enhancer",
            description=(
                "Enhance dark/bright local regions for forensic observation using CLAHE-style "
                "adaptive contrast/brightness adjustment."
            ),
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "image_input": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Local image path.",
                    ),
                    "image_path": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Alias of image_input.",
                    ),
                    "method": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Enhancement method. Currently supports clahe.",
                        enum=["clahe"],
                    ),
                    "clip_limit": OpenAIFunctionPropertySchema(
                        type="number",
                        description="CLAHE clip limit multiplier, typical range 1.0-6.0.",
                    ),
                    "tile_grid_size": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="CLAHE tile grid size per side, range 2-16.",
                    ),
                    "brightness_gain": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Post brightness gain on luma, range 0.5-2.0.",
                    ),
                    "contrast_gain": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Post contrast gain on luma, range 0.5-2.0.",
                    ),
                    "gamma": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Gamma correction on luma, range 0.5-2.5.",
                    ),
                    "output_format": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Output format: PNG|JPEG|WEBP.",
                        enum=["PNG", "JPEG", "WEBP"],
                    ),
                    "jpeg_quality": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="JPEG quality when output_format=JPEG, range 60-100.",
                    ),
                    "output_dir": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Directory to save enhanced image and artifact map.",
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


def _clahe_lut(tile_u8: np.ndarray, clip_limit: float) -> np.ndarray:
    hist = np.bincount(tile_u8.ravel(), minlength=256).astype(np.float32)
    tile_size = float(tile_u8.size)
    clip_abs = max(1.0, clip_limit * tile_size / 256.0)

    excess = np.maximum(hist - clip_abs, 0.0)
    hist = np.minimum(hist, clip_abs)
    redist = float(np.sum(excess)) / 256.0
    hist += redist

    cdf = np.cumsum(hist)
    if cdf[-1] <= 1e-8:
        return np.arange(256, dtype=np.uint8)

    cdf_norm = (cdf - cdf[0]) / (cdf[-1] - cdf[0] + 1e-8)
    lut = np.clip(np.round(cdf_norm * 255.0), 0.0, 255.0).astype(np.uint8)
    return lut


def _apply_clahe_u8(img_u8: np.ndarray, tile_grid_size: int, clip_limit: float) -> np.ndarray:
    h, w = img_u8.shape
    g = max(2, tile_grid_size)

    tile_h = int(np.ceil(float(h) / float(g)))
    tile_w = int(np.ceil(float(w) / float(g)))

    pad_h = tile_h * g
    pad_w = tile_w * g

    padded = np.pad(img_u8, ((0, pad_h - h), (0, pad_w - w)), mode="edge")

    luts = np.zeros((g, g, 256), dtype=np.uint8)
    for gy in range(g):
        for gx in range(g):
            y0 = gy * tile_h
            y1 = y0 + tile_h
            x0 = gx * tile_w
            x1 = x0 + tile_w
            tile = padded[y0:y1, x0:x1]
            luts[gy, gx] = _clahe_lut(tile, clip_limit=clip_limit)

    out = np.zeros((pad_h, pad_w), dtype=np.float32)

    for y in range(pad_h):
        fy = (float(y) + 0.5) / float(tile_h) - 0.5
        y0 = int(np.floor(fy))
        y1 = min(y0 + 1, g - 1)
        y0 = max(0, min(g - 1, y0))
        wy = fy - float(y0)
        wy = _clip(wy, 0.0, 1.0)

        for x in range(pad_w):
            fx = (float(x) + 0.5) / float(tile_w) - 0.5
            x0 = int(np.floor(fx))
            x1 = min(x0 + 1, g - 1)
            x0 = max(0, min(g - 1, x0))
            wx = fx - float(x0)
            wx = _clip(wx, 0.0, 1.0)

            pv = int(padded[y, x])
            v00 = float(luts[y0, x0, pv])
            v01 = float(luts[y0, x1, pv])
            v10 = float(luts[y1, x0, pv])
            v11 = float(luts[y1, x1, pv])

            top = v00 * (1.0 - wx) + v01 * wx
            bot = v10 * (1.0 - wx) + v11 * wx
            out[y, x] = top * (1.0 - wy) + bot * wy

    out = np.clip(np.round(out[:h, :w]), 0.0, 255.0).astype(np.uint8)
    return out


class VisualEnhancer:
    def __init__(
        self,
        default_method: str = "clahe",
        default_clip_limit: float = 2.0,
        default_tile_grid_size: int = 8,
        default_brightness_gain: float = 1.0,
        default_contrast_gain: float = 1.0,
        default_gamma: float = 1.0,
        default_output_format: str = "PNG",
        default_jpeg_quality: int = 95,
        default_output_dir: str = "./outputs/visual_enhancer",
    ):
        self.tool_name = "Visual_Enhancer"
        self.default_method = default_method
        self.default_clip_limit = default_clip_limit
        self.default_tile_grid_size = default_tile_grid_size
        self.default_brightness_gain = default_brightness_gain
        self.default_contrast_gain = default_contrast_gain
        self.default_gamma = default_gamma
        self.default_output_format = default_output_format
        self.default_jpeg_quality = default_jpeg_quality
        self.default_output_dir = default_output_dir

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

    def _enhance_luma(
        self,
        y_u8: np.ndarray,
        method: str,
        clip_limit: float,
        tile_grid_size: int,
        brightness_gain: float,
        contrast_gain: float,
        gamma: float,
    ) -> np.ndarray:
        if method != "clahe":
            method = "clahe"

        y_enh = _apply_clahe_u8(y_u8, tile_grid_size=tile_grid_size, clip_limit=clip_limit)

        y_f = y_enh.astype(np.float32) / 255.0
        # contrast around center and brightness gain.
        y_f = (y_f - 0.5) * contrast_gain + 0.5
        y_f = y_f * brightness_gain
        y_f = np.clip(y_f, 0.0, 1.0)

        # gamma correction on enhanced luminance.
        y_f = np.power(y_f, 1.0 / gamma)
        y_out = np.clip(np.round(y_f * 255.0), 0.0, 255.0).astype(np.uint8)
        return y_out

    def __call__(self, args_str: str) -> Dict[str, Any]:
        start_time = time.time()
        log_call_args: Dict[str, Any] = {"raw_args": args_str}

        def _finalize(response_obj: Dict[str, Any]) -> Dict[str, Any]:
            log_tool_use_record(
                tool_key="visual_enhancer",
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
                    description="Visual enhancement failed.",
                    error_message="工具参数解析失败，必须是有效的 JSON 格式字符串。",
                )
            )

        image_path = args.get("image_path") or args.get("image_input")
        if not image_path:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Visual enhancement failed.",
                    error_message="缺少必填参数 image_path。",
                )
            )
        if not os.path.exists(image_path):
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Visual enhancement failed.",
                    error_message=f"找不到图像文件: {image_path}",
                )
            )

        method = str(args.get("method", self.default_method)).strip().lower()
        if method not in {"clahe"}:
            method = "clahe"

        clip_limit = _clip(_safe_float(args.get("clip_limit", self.default_clip_limit), self.default_clip_limit), 1.0, 6.0)
        tile_grid_size = _safe_int(args.get("tile_grid_size", self.default_tile_grid_size), self.default_tile_grid_size)
        tile_grid_size = max(2, min(16, tile_grid_size))

        brightness_gain = _clip(
            _safe_float(args.get("brightness_gain", self.default_brightness_gain), self.default_brightness_gain),
            0.5,
            2.0,
        )
        contrast_gain = _clip(
            _safe_float(args.get("contrast_gain", self.default_contrast_gain), self.default_contrast_gain),
            0.5,
            2.0,
        )
        gamma = _clip(_safe_float(args.get("gamma", self.default_gamma), self.default_gamma), 0.5, 2.5)

        output_format = str(args.get("output_format", self.default_output_format)).upper().strip()
        if output_format not in {"PNG", "JPEG", "WEBP"}:
            output_format = "PNG"
        jpeg_quality = _safe_int(args.get("jpeg_quality", self.default_jpeg_quality), self.default_jpeg_quality)
        jpeg_quality = max(60, min(100, jpeg_quality))
        output_dir = str(args.get("output_dir", self.default_output_dir))

        log_call_args = {
            "image_path": image_path,
            "method": method,
            "clip_limit": clip_limit,
            "tile_grid_size": tile_grid_size,
            "brightness_gain": brightness_gain,
            "contrast_gain": contrast_gain,
            "gamma": gamma,
            "output_format": output_format,
            "jpeg_quality": jpeg_quality,
            "output_dir": output_dir,
        }

        try:
            rgb = Image.open(image_path).convert("RGB")
            ycbcr = rgb.convert("YCbCr")
            # np.asarray(PIL.Image) may return a read-only view; take a writable copy.
            ycbcr_arr = np.array(ycbcr, dtype=np.uint8, copy=True)
            y_u8 = ycbcr_arr[:, :, 0]
        except Exception as exc:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Visual enhancement failed.",
                    error_message=f"图像读取失败: {exc}",
                )
            )

        try:
            y_enh = self._enhance_luma(
                y_u8=y_u8,
                method=method,
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size,
                brightness_gain=brightness_gain,
                contrast_gain=contrast_gain,
                gamma=gamma,
            )
        except Exception as exc:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Visual enhancement failed.",
                    error_message=f"增强处理失败: {exc}",
                )
            )

        ycbcr_arr[:, :, 0] = y_enh
        enhanced = Image.fromarray(ycbcr_arr, mode="YCbCr").convert("RGB")

        # artifact visibility map: local luminance change.
        vis_map = np.abs(y_enh.astype(np.float32) - y_u8.astype(np.float32))
        vmax = float(np.percentile(vis_map, 99.0))
        if vmax < 1e-6:
            vis_u8 = np.zeros_like(y_u8, dtype=np.uint8)
        else:
            vis_u8 = np.clip(np.round(vis_map / vmax * 255.0), 0.0, 255.0).astype(np.uint8)
        vis_img = Image.fromarray(vis_u8, mode="L")

        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        enhanced_path = os.path.join(output_dir, f"{stem}_enhanced_{method}.{output_format.lower()}")
        vis_path = os.path.join(output_dir, f"{stem}_artifact_visibility_map.png")

        try:
            if output_format == "JPEG":
                enhanced.save(enhanced_path, format="JPEG", quality=jpeg_quality)
            else:
                enhanced.save(enhanced_path, format=output_format)
            vis_img.save(vis_path, format="PNG")
        except Exception as exc:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Visual enhancement failed.",
                    error_message=f"结果写出失败: {exc}",
                )
            )

        dark_ratio_before = float(np.mean(y_u8 <= 32))
        dark_ratio_after = float(np.mean(y_enh <= 32))
        bright_ratio_before = float(np.mean(y_u8 >= 223))
        bright_ratio_after = float(np.mean(y_enh >= 223))

        dr_before = float(np.percentile(y_u8, 95.0) - np.percentile(y_u8, 5.0))
        dr_after = float(np.percentile(y_enh, 95.0) - np.percentile(y_enh, 5.0))

        forensic_hint = (
            "已增强极暗/极亮区域局部对比度，可用于观察可能被光照伪装的拼接边缘与纹理异常。"
            if dr_after > dr_before
            else "动态范围提升有限，建议提高 clip_limit 或 tile_grid_size 后重试。"
        )

        result = {
            "enhancement_report": {
                "method": method,
                "clip_limit": round(clip_limit, 4),
                "tile_grid_size": int(tile_grid_size),
                "brightness_gain": round(brightness_gain, 4),
                "contrast_gain": round(contrast_gain, 4),
                "gamma": round(gamma, 4),
                "dynamic_range_before": round(dr_before, 4),
                "dynamic_range_after": round(dr_after, 4),
                "dark_ratio_before": round(dark_ratio_before, 6),
                "dark_ratio_after": round(dark_ratio_after, 6),
                "bright_ratio_before": round(bright_ratio_before, 6),
                "bright_ratio_after": round(bright_ratio_after, 6),
                "forensic_hint": forensic_hint,
                "summary": "局部自适应对比度/亮度增强完成，已输出增强图与伪影可见性图。",
            },
            "artifacts": {
                "enhanced_image_path": enhanced_path,
                "artifact_visibility_map_path": vis_path,
            },
        }

        return _finalize(
            self._build_response(
                success=True,
                start_time=start_time,
                description="Visual enhancement completed.",
                result=result,
            )
        )


class VisualEnhancerTool(BaseTool):
    """verl-native wrapper aligned with BaseTool lifecycle."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema or _default_schema())
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self._default_method = str(config.get("default_method", "clahe"))
        self._default_clip_limit = float(config.get("default_clip_limit", 2.0))
        self._default_tile_grid_size = int(config.get("default_tile_grid_size", 8))
        self._default_brightness_gain = float(config.get("default_brightness_gain", 1.0))
        self._default_contrast_gain = float(config.get("default_contrast_gain", 1.0))
        self._default_gamma = float(config.get("default_gamma", 1.0))
        self._default_output_format = str(config.get("default_output_format", "PNG"))
        self._default_jpeg_quality = int(config.get("default_jpeg_quality", 95))
        self._default_output_dir = str(config.get("output_dir", "./outputs/visual_enhancer"))
        self._tool_impl = VisualEnhancer(
            default_method=self._default_method,
            default_clip_limit=self._default_clip_limit,
            default_tile_grid_size=self._default_tile_grid_size,
            default_brightness_gain=self._default_brightness_gain,
            default_contrast_gain=self._default_contrast_gain,
            default_gamma=self._default_gamma,
            default_output_format=self._default_output_format,
            default_jpeg_quality=self._default_jpeg_quality,
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
            call_args: Dict[str, Any] = dict(parameters)
            call_args["image_path"] = (
                kwargs.get("image_input")
                or kwargs.get("image_path")
                or parameters.get("image_input")
                or parameters.get("image_path")
                or default_image_path
            )
            call_args.setdefault("method", self._default_method)
            call_args.setdefault("clip_limit", self._default_clip_limit)
            call_args.setdefault("tile_grid_size", self._default_tile_grid_size)
            call_args.setdefault("brightness_gain", self._default_brightness_gain)
            call_args.setdefault("contrast_gain", self._default_contrast_gain)
            call_args.setdefault("gamma", self._default_gamma)
            call_args.setdefault("output_format", self._default_output_format)
            call_args.setdefault("jpeg_quality", self._default_jpeg_quality)
            call_args.setdefault("output_dir", self._default_output_dir)

            result = self._tool_impl(json.dumps(call_args, ensure_ascii=False))
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["response"] = result
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as exc:
            logger.exception("VisualEnhancerTool execution failed.")
            error_result = {
                "success": False,
                "status": "error",
                "description": "Visual enhancement failed.",
                "result": {},
                "error_message": f"Tool runtime error: {exc}",
            }
            return ToolResponse(text=json.dumps(error_result, ensure_ascii=False)), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
