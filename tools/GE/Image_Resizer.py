import json
import logging
import os
import time
from typing import Any, Dict, Optional
from uuid import uuid4

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


_INTERPOLATION_MAP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


def _default_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="image_resizer",
            description=(
                "Resize image with strict interpolation control and record scale factor metadata "
                "for forensic downstream sensitivity tracking."
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
                    "target_width": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Target width in pixels. If absent, infer from scale_factor.",
                    ),
                    "target_height": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Target height in pixels. If absent, infer from scale_factor.",
                    ),
                    "scale_factor": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Uniform resize factor s (>0).",
                    ),
                    "keep_aspect_ratio": OpenAIFunctionPropertySchema(
                        type="boolean",
                        description="If true and only one target side is provided, infer the other side.",
                    ),
                    "interpolation": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Interpolation algorithm: nearest|bilinear|bicubic|lanczos.",
                        enum=["nearest", "bilinear", "bicubic", "lanczos"],
                    ),
                    "prefer_forensics_mode": OpenAIFunctionPropertySchema(
                        type="string",
                        description=(
                            "forensics helper mode: auto|pixel_inspection|model_adaptation. "
                            "pixel_inspection prefers nearest."
                        ),
                        enum=["auto", "pixel_inspection", "model_adaptation"],
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
                        description="Directory to save resized image.",
                    ),
                },
                required=["image_input"],
            ),
            strict=False,
        ),
    )


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


class ImageResizer:
    def __init__(
        self,
        default_interpolation: str = "nearest",
        default_prefer_forensics_mode: str = "auto",
        default_keep_aspect_ratio: bool = True,
        default_output_format: str = "PNG",
        default_jpeg_quality: int = 95,
        default_output_dir: str = "./outputs/image_resizer",
    ):
        self.tool_name = "Image_Resizer"
        self.default_interpolation = default_interpolation
        self.default_prefer_forensics_mode = default_prefer_forensics_mode
        self.default_keep_aspect_ratio = default_keep_aspect_ratio
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

    def _resolve_interpolation(self, interpolation: str, mode: str) -> str:
        interp = interpolation.strip().lower()
        if interp not in _INTERPOLATION_MAP:
            interp = self.default_interpolation

        fmode = mode.strip().lower()
        if fmode == "pixel_inspection":
            return "nearest"
        if fmode == "model_adaptation" and interp == "nearest":
            # Preserve explicit nearest only if user asked it.
            return interp
        return interp

    def _compute_target_size(
        self,
        src_w: int,
        src_h: int,
        target_w: Optional[int],
        target_h: Optional[int],
        scale_factor: Optional[float],
        keep_aspect_ratio: bool,
    ) -> Dict[str, Any]:
        if scale_factor is not None:
            s = max(1e-6, float(scale_factor))
            tw = max(1, int(round(src_w * s)))
            th = max(1, int(round(src_h * s)))
            return {"target_width": tw, "target_height": th, "sx": s, "sy": s, "mode": "scale_factor"}

        has_w = target_w is not None and target_w > 0
        has_h = target_h is not None and target_h > 0

        if not has_w and not has_h:
            raise ValueError("必须提供 scale_factor 或 target_width/target_height。")

        if keep_aspect_ratio:
            if has_w and not has_h:
                tw = int(target_w)
                th = max(1, int(round(src_h * (float(tw) / float(src_w)))))
            elif has_h and not has_w:
                th = int(target_h)
                tw = max(1, int(round(src_w * (float(th) / float(src_h)))))
            else:
                tw = int(target_w)
                th = int(target_h)
        else:
            tw = int(target_w) if has_w else src_w
            th = int(target_h) if has_h else src_h

        tw = max(1, tw)
        th = max(1, th)
        sx = float(tw) / float(src_w)
        sy = float(th) / float(src_h)
        return {"target_width": tw, "target_height": th, "sx": sx, "sy": sy, "mode": "target_size"}

    def __call__(self, args_str: str) -> Dict[str, Any]:
        start_time = time.time()
        log_call_args: Dict[str, Any] = {"raw_args": args_str}

        def _finalize(response_obj: Dict[str, Any]) -> Dict[str, Any]:
            log_tool_use_record(
                tool_key="image_resizer",
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
                    description="Image resize failed.",
                    error_message="工具参数解析失败，必须是有效的 JSON 格式字符串。",
                )
            )

        image_path = args.get("image_path") or args.get("image_input")
        if not image_path:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Image resize failed.",
                    error_message="缺少必填参数 image_path。",
                )
            )
        if not os.path.exists(image_path):
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Image resize failed.",
                    error_message=f"找不到图像文件: {image_path}",
                )
            )

        keep_aspect_ratio = bool(args.get("keep_aspect_ratio", self.default_keep_aspect_ratio))
        mode = str(args.get("prefer_forensics_mode", self.default_prefer_forensics_mode)).strip().lower()
        if mode not in {"auto", "pixel_inspection", "model_adaptation"}:
            mode = "auto"

        interpolation_req = str(args.get("interpolation", self.default_interpolation)).strip().lower()
        interpolation = self._resolve_interpolation(interpolation_req, mode)

        output_format = str(args.get("output_format", self.default_output_format)).upper().strip()
        if output_format not in {"PNG", "JPEG", "WEBP"}:
            output_format = "PNG"

        jpeg_quality = _safe_int(args.get("jpeg_quality", self.default_jpeg_quality), self.default_jpeg_quality)
        jpeg_quality = max(60, min(100, jpeg_quality))
        output_dir = str(args.get("output_dir", self.default_output_dir))

        target_w = args.get("target_width")
        target_h = args.get("target_height")
        target_w = _safe_int(target_w, -1) if target_w is not None else None
        target_h = _safe_int(target_h, -1) if target_h is not None else None

        scale_factor = args.get("scale_factor")
        sf = _safe_float(scale_factor, -1.0) if scale_factor is not None else None
        if sf is not None and sf <= 0:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Image resize failed.",
                    error_message="scale_factor 必须大于 0。",
                )
            )

        log_call_args = {
            "image_path": image_path,
            "target_width": target_w,
            "target_height": target_h,
            "scale_factor": sf,
            "keep_aspect_ratio": keep_aspect_ratio,
            "interpolation": interpolation,
            "prefer_forensics_mode": mode,
            "output_format": output_format,
            "jpeg_quality": jpeg_quality,
            "output_dir": output_dir,
        }

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Image resize failed.",
                    error_message=f"图像读取失败: {exc}",
                )
            )

        src_w, src_h = image.size

        try:
            size_info = self._compute_target_size(
                src_w=src_w,
                src_h=src_h,
                target_w=target_w,
                target_h=target_h,
                scale_factor=sf,
                keep_aspect_ratio=keep_aspect_ratio,
            )
        except Exception as exc:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Image resize failed.",
                    error_message=f"目标尺寸计算失败: {exc}",
                )
            )

        tw = int(size_info["target_width"])
        th = int(size_info["target_height"])
        sx = float(size_info["sx"])
        sy = float(size_info["sy"])

        try:
            resized = image.resize((tw, th), _INTERPOLATION_MAP[interpolation])
        except Exception as exc:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Image resize failed.",
                    error_message=f"缩放失败: {exc}",
                )
            )

        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(
            output_dir,
            f"{stem}_resized_{tw}x{th}_{interpolation}.{output_format.lower()}",
        )

        try:
            if output_format == "JPEG":
                resized.save(out_path, format="JPEG", quality=jpeg_quality)
            else:
                resized.save(out_path, format=output_format)
        except Exception as exc:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="Image resize failed.",
                    error_message=f"结果写出失败: {exc}",
                )
            )

        # forensic sensitivity hint for resampling impact on pixel-level tools.
        if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
            sensitivity_hint = "未发生缩放，像素统计敏感度基本不变。"
        elif sx > 1.0 or sy > 1.0:
            if interpolation == "nearest":
                sensitivity_hint = "已放大且使用最近邻，像素值未被平滑，适合像素级检查。"
            else:
                sensitivity_hint = "已放大且使用平滑插值，可能改变细粒度取证特征。"
        else:
            sensitivity_hint = "已缩小（重采样），请在后续取证工具中结合缩放因子 s 解释结果。"

        result = {
            "resize_report": {
                "original_size": {"width": int(src_w), "height": int(src_h)},
                "target_size": {"width": int(tw), "height": int(th)},
                "scale_factor": {
                    "sx": round(sx, 6),
                    "sy": round(sy, 6),
                    "s": round((sx + sy) * 0.5, 6),
                    "is_uniform": bool(abs(sx - sy) < 1e-6),
                },
                "interpolation": interpolation,
                "prefer_forensics_mode": mode,
                "resampling_applied": bool(abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6),
                "sensitivity_hint": sensitivity_hint,
                "summary": "图像缩放完成，并记录了缩放因子与插值信息供后续取证链路使用。",
            },
            "artifacts": {
                "resized_image_path": out_path,
            },
        }

        return _finalize(
            self._build_response(
                success=True,
                start_time=start_time,
                description="Image resize completed.",
                result=result,
            )
        )


class ImageResizerTool(BaseTool):
    """verl-native wrapper aligned with BaseTool lifecycle."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema or _default_schema())
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self._default_interpolation = str(config.get("default_interpolation", "nearest"))
        self._default_prefer_forensics_mode = str(config.get("default_prefer_forensics_mode", "auto"))
        self._default_keep_aspect_ratio = bool(config.get("default_keep_aspect_ratio", True))
        self._default_output_format = str(config.get("default_output_format", "PNG"))
        self._default_jpeg_quality = int(config.get("default_jpeg_quality", 95))
        self._default_output_dir = str(config.get("output_dir", "./outputs/image_resizer"))
        self._tool_impl = ImageResizer(
            default_interpolation=self._default_interpolation,
            default_prefer_forensics_mode=self._default_prefer_forensics_mode,
            default_keep_aspect_ratio=self._default_keep_aspect_ratio,
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
            call_args.setdefault("interpolation", self._default_interpolation)
            call_args.setdefault("prefer_forensics_mode", self._default_prefer_forensics_mode)
            call_args.setdefault("keep_aspect_ratio", self._default_keep_aspect_ratio)
            call_args.setdefault("output_format", self._default_output_format)
            call_args.setdefault("jpeg_quality", self._default_jpeg_quality)
            call_args.setdefault("output_dir", self._default_output_dir)

            result = self._tool_impl(json.dumps(call_args, ensure_ascii=False))
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["response"] = result
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as exc:
            logger.exception("ImageResizerTool execution failed.")
            error_result = {
                "success": False,
                "status": "error",
                "description": "Image resize failed.",
                "result": {},
                "error_message": f"Tool runtime error: {exc}",
            }
            return ToolResponse(text=json.dumps(error_result, ensure_ascii=False)), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
