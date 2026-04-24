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
            name="roi_extractor",
            description=(
                "Extract ROI sub-image and backtrack local mask to original global coordinates. "
                "Useful for large-image forensic acceleration and coordinate tracing."
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
                    "roi": OpenAIFunctionPropertySchema(
                        type="object",
                        description=(
                            "ROI box object: {x,y,w,h,coord_type,coordinate_space,source_width,source_height}. "
                            "coord_type supports abs/relative. coordinate_space supports original/scaled."
                        ),
                    ),
                    "x": OpenAIFunctionPropertySchema(type="number", description="ROI x (top-left)."),
                    "y": OpenAIFunctionPropertySchema(type="number", description="ROI y (top-left)."),
                    "w": OpenAIFunctionPropertySchema(type="number", description="ROI width."),
                    "h": OpenAIFunctionPropertySchema(type="number", description="ROI height."),
                    "coord_type": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Coordinate type for ROI fields: abs or relative.",
                        enum=["abs", "relative"],
                    ),
                    "coordinate_space": OpenAIFunctionPropertySchema(
                        type="string",
                        description="ROI coordinate space: original or scaled.",
                        enum=["original", "scaled"],
                    ),
                    "source_width": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Required when coordinate_space=scaled. Width used by detector stage.",
                    ),
                    "source_height": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Required when coordinate_space=scaled. Height used by detector stage.",
                    ),
                    "mask_input": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Optional local mask path aligned with ROI or ROI-resized image.",
                    ),
                    "local_ref_width": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Optional reference width for local mask coordinate system.",
                    ),
                    "local_ref_height": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Optional reference height for local mask coordinate system.",
                    ),
                    "mask_threshold": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Binarization threshold for mask in [0,1], default 0.5.",
                    ),
                    "export_crop": OpenAIFunctionPropertySchema(
                        type="boolean",
                        description="Whether to export ROI crop image.",
                    ),
                    "export_global_mask": OpenAIFunctionPropertySchema(
                        type="boolean",
                        description="Whether to export backtracked global mask.",
                    ),
                    "output_dir": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Output directory for exported files.",
                    ),
                },
                required=["image_input"],
            ),
            strict=False,
        ),
    )


def _clamp_float(v: Any, low: float, high: float, default: float) -> float:
    try:
        f = float(v)
    except Exception:
        f = default
    return max(low, min(high, f))


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default


class ROIExtractor:
    def __init__(
        self,
        default_export_crop: bool = True,
        default_export_global_mask: bool = True,
        default_output_dir: str = "./outputs/roi_extractor",
        default_mask_threshold: float = 0.5,
    ):
        self.tool_name = "ROI_Extractor"
        self.default_export_crop = default_export_crop
        self.default_export_global_mask = default_export_global_mask
        self.default_output_dir = default_output_dir
        self.default_mask_threshold = default_mask_threshold

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

    def _parse_roi(self, args: Dict[str, Any], img_w: int, img_h: int) -> Dict[str, int]:
        roi_obj = args.get("roi", {})
        if not isinstance(roi_obj, dict):
            roi_obj = {}

        x = roi_obj.get("x", args.get("x"))
        y = roi_obj.get("y", args.get("y"))
        w = roi_obj.get("w", args.get("w"))
        h = roi_obj.get("h", args.get("h"))

        if x is None or y is None or w is None or h is None:
            raise ValueError("缺少 ROI 参数，必须提供 x,y,w,h 或 roi 对象。")

        coord_type = str(roi_obj.get("coord_type", args.get("coord_type", "abs"))).strip().lower()
        if coord_type not in {"abs", "relative"}:
            coord_type = "abs"

        coordinate_space = str(roi_obj.get("coordinate_space", args.get("coordinate_space", "original"))).strip().lower()
        if coordinate_space not in {"original", "scaled"}:
            coordinate_space = "original"

        src_w = _safe_int(roi_obj.get("source_width", args.get("source_width", img_w)), img_w)
        src_h = _safe_int(roi_obj.get("source_height", args.get("source_height", img_h)), img_h)
        if src_w <= 0:
            src_w = img_w
        if src_h <= 0:
            src_h = img_h

        if coord_type == "relative":
            if coordinate_space == "scaled":
                bx, by = src_w, src_h
            else:
                bx, by = img_w, img_h
            rx = float(x) * float(bx)
            ry = float(y) * float(by)
            rw = float(w) * float(bx)
            rh = float(h) * float(by)
        else:
            rx = float(x)
            ry = float(y)
            rw = float(w)
            rh = float(h)

        if coordinate_space == "scaled":
            sx = float(img_w) / float(src_w)
            sy = float(img_h) / float(src_h)
            rx *= sx
            ry *= sy
            rw *= sx
            rh *= sy

        x0 = max(0, min(img_w - 1, int(round(rx))))
        y0 = max(0, min(img_h - 1, int(round(ry))))
        ww = max(1, int(round(rw)))
        hh = max(1, int(round(rh)))
        x1 = min(img_w, x0 + ww)
        y1 = min(img_h, y0 + hh)

        if x1 <= x0 or y1 <= y0:
            raise ValueError("ROI 落在图像外部或宽高无效。")

        return {
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "w": x1 - x0,
            "h": y1 - y0,
        }

    def _backtrack_mask(
        self,
        mask_path: str,
        roi_box: Dict[str, int],
        img_w: int,
        img_h: int,
        local_ref_w: Optional[int],
        local_ref_h: Optional[int],
        threshold: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        mask_img = Image.open(mask_path).convert("L")
        local_w, local_h = mask_img.size

        if local_ref_w is None or local_ref_w <= 0:
            local_ref_w = local_w
        if local_ref_h is None or local_ref_h <= 0:
            local_ref_h = local_h

        # If mask is not in declared local reference size, align it first.
        if local_w != local_ref_w or local_h != local_ref_h:
            mask_img = mask_img.resize((local_ref_w, local_ref_h), Image.BILINEAR)

        # Map local mask space to ROI space on original image.
        roi_w, roi_h = roi_box["w"], roi_box["h"]
        mask_roi = mask_img.resize((roi_w, roi_h), Image.BILINEAR)
        mask_arr = np.asarray(mask_roi, dtype=np.float32) / 255.0
        binary = (mask_arr >= threshold).astype(np.uint8)

        global_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        x0, y0 = roi_box["x0"], roi_box["y0"]
        x1, y1 = roi_box["x1"], roi_box["y1"]
        global_mask[y0:y1, x0:x1] = binary

        ys, xs = np.where(global_mask > 0)
        bbox = None
        if ys.size > 0:
            bbox = {
                "x0": int(np.min(xs)),
                "y0": int(np.min(ys)),
                "x1": int(np.max(xs) + 1),
                "y1": int(np.max(ys) + 1),
                "w": int(np.max(xs) - np.min(xs) + 1),
                "h": int(np.max(ys) - np.min(ys) + 1),
                "area": int(ys.size),
            }

        transform = {
            "global_offset_x": int(x0),
            "global_offset_y": int(y0),
            "local_ref_width": int(local_ref_w),
            "local_ref_height": int(local_ref_h),
            "roi_width": int(roi_w),
            "roi_height": int(roi_h),
            "scale_x_local_to_global": float(roi_w) / float(local_ref_w),
            "scale_y_local_to_global": float(roi_h) / float(local_ref_h),
            "mask_threshold": float(threshold),
        }

        return global_mask, {"global_mask_bbox": bbox, "mapping_transform": transform}

    def __call__(self, args_str: str) -> Dict[str, Any]:
        start_time = time.time()
        log_call_args: Dict[str, Any] = {"raw_args": args_str}

        def _finalize(response_obj: Dict[str, Any]) -> Dict[str, Any]:
            log_tool_use_record(
                tool_key="roi_extractor",
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
                    description="ROI extraction failed.",
                    error_message="工具参数解析失败，必须是有效的 JSON 格式字符串。",
                )
            )

        image_path = args.get("image_path") or args.get("image_input")
        if not image_path:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="ROI extraction failed.",
                    error_message="缺少必填参数 image_path。",
                )
            )
        if not os.path.exists(image_path):
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="ROI extraction failed.",
                    error_message=f"找不到图像文件: {image_path}",
                )
            )

        export_crop = bool(args.get("export_crop", self.default_export_crop))
        export_global_mask = bool(args.get("export_global_mask", self.default_export_global_mask))
        output_dir = str(args.get("output_dir", self.default_output_dir))
        mask_input = args.get("mask_input")
        threshold = _clamp_float(args.get("mask_threshold", self.default_mask_threshold), 0.0, 1.0, self.default_mask_threshold)

        local_ref_w = args.get("local_ref_width")
        local_ref_h = args.get("local_ref_height")
        local_ref_w = _safe_int(local_ref_w, -1) if local_ref_w is not None else None
        local_ref_h = _safe_int(local_ref_h, -1) if local_ref_h is not None else None

        log_call_args = {
            "image_path": image_path,
            "export_crop": export_crop,
            "export_global_mask": export_global_mask,
            "output_dir": output_dir,
            "mask_input": mask_input,
            "mask_threshold": threshold,
        }

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="ROI extraction failed.",
                    error_message=f"图像读取失败: {exc}",
                )
            )

        img_w, img_h = image.size

        try:
            roi_box = self._parse_roi(args=args, img_w=img_w, img_h=img_h)
        except Exception as exc:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="ROI extraction failed.",
                    error_message=f"ROI 参数错误: {exc}",
                )
            )

        x0, y0, x1, y1 = roi_box["x0"], roi_box["y0"], roi_box["x1"], roi_box["y1"]
        roi_img = image.crop((x0, y0, x1, y1))

        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]

        roi_path = ""
        if export_crop:
            roi_path = os.path.join(output_dir, f"{stem}_roi_x{x0}_y{y0}_w{roi_box['w']}_h{roi_box['h']}.png")
            roi_img.save(roi_path)

        global_mask_path = ""
        mask_info: Dict[str, Any] = {
            "global_mask_bbox": None,
            "mapping_transform": None,
        }

        if mask_input:
            if not os.path.exists(mask_input):
                return _finalize(
                    self._build_response(
                        success=False,
                        start_time=start_time,
                        description="ROI extraction failed.",
                        error_message=f"mask_input 不存在: {mask_input}",
                    )
                )
            try:
                global_mask, mask_info = self._backtrack_mask(
                    mask_path=str(mask_input),
                    roi_box=roi_box,
                    img_w=img_w,
                    img_h=img_h,
                    local_ref_w=local_ref_w,
                    local_ref_h=local_ref_h,
                    threshold=threshold,
                )
                if export_global_mask:
                    global_mask_path = os.path.join(output_dir, f"{stem}_global_mask.png")
                    Image.fromarray((global_mask * 255).astype(np.uint8), mode="L").save(global_mask_path)
            except Exception as exc:
                return _finalize(
                    self._build_response(
                        success=False,
                        start_time=start_time,
                        description="ROI extraction failed.",
                        error_message=f"掩码回溯失败: {exc}",
                    )
                )

        result = {
            "roi_result": {
                "global_roi_box": roi_box,
                "global_roi_box_relative": {
                    "x": round(float(x0) / float(img_w), 6),
                    "y": round(float(y0) / float(img_h), 6),
                    "w": round(float(roi_box["w"]) / float(img_w), 6),
                    "h": round(float(roi_box["h"]) / float(img_h), 6),
                },
                "original_size": {"width": int(img_w), "height": int(img_h)},
                "roi_size": {"width": int(roi_box["w"]), "height": int(roi_box["h"])},
                "summary": "ROI 裁剪完成，若提供局部掩码则已执行坐标回溯到原图全局坐标系。",
            },
            "mask_backtracking": {
                "has_local_mask_input": bool(mask_input),
                "global_mask_bbox": mask_info.get("global_mask_bbox"),
                "mapping_transform": mask_info.get("mapping_transform"),
            },
            "artifacts": {
                "roi_crop_path": roi_path,
                "global_mask_path": global_mask_path,
            },
        }

        return _finalize(
            self._build_response(
                success=True,
                start_time=start_time,
                description="ROI extraction and coordinate backtracking completed.",
                result=result,
            )
        )


class ROIExtractorTool(BaseTool):
    """verl-native wrapper aligned with BaseTool lifecycle."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema or _default_schema())
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self._default_export_crop = bool(config.get("default_export_crop", True))
        self._default_export_global_mask = bool(config.get("default_export_global_mask", True))
        self._default_output_dir = str(config.get("output_dir", "./outputs/roi_extractor"))
        self._default_mask_threshold = float(config.get("default_mask_threshold", 0.5))
        self._tool_impl = ROIExtractor(
            default_export_crop=self._default_export_crop,
            default_export_global_mask=self._default_export_global_mask,
            default_output_dir=self._default_output_dir,
            default_mask_threshold=self._default_mask_threshold,
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
            call_args.setdefault("export_crop", self._default_export_crop)
            call_args.setdefault("export_global_mask", self._default_export_global_mask)
            call_args.setdefault("output_dir", self._default_output_dir)
            call_args.setdefault("mask_threshold", self._default_mask_threshold)

            result = self._tool_impl(json.dumps(call_args, ensure_ascii=False))
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["response"] = result
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as exc:
            logger.exception("ROIExtractorTool execution failed.")
            error_result = {
                "success": False,
                "status": "error",
                "description": "ROI extraction failed.",
                "result": {},
                "error_message": f"Tool runtime error: {exc}",
            }
            return ToolResponse(text=json.dumps(error_result, ensure_ascii=False)), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
