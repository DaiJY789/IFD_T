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


try:
    import mediapipe as mp  # type: ignore

    _HAS_MEDIAPIPE = True
except Exception:
    mp = None  # type: ignore
    _HAS_MEDIAPIPE = False

try:
    from insightface.app import FaceAnalysis  # type: ignore

    _HAS_INSIGHTFACE = True
except Exception:
    FaceAnalysis = None  # type: ignore
    _HAS_INSIGHTFACE = False


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _default_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="facial_biological_auditor",
            description=(
                "Audit face biological/physical consistency for deepfake detection using facial landmarks, "
                "corneal reflection geometry, and second-order texture checks on iris/teeth regions."
            ),
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "image_input": OpenAIFunctionPropertySchema(type="string", description="Local image path."),
                    "image_path": OpenAIFunctionPropertySchema(type="string", description="Alias of image_input."),
                    "keypoint_backend": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Landmark backend: auto|mediapipe|insightface.",
                        enum=["auto", "mediapipe", "insightface"],
                    ),
                    "max_side": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Max image side for speed, range 256-1024.",
                    ),
                    "reflection_angle_threshold_deg": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Corneal reflection mirrored-angle inconsistency threshold in degree.",
                    ),
                    "reflection_mag_ratio_threshold": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Corneal reflection vector magnitude ratio threshold.",
                    ),
                    "texture_blur_ratio_threshold": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Second-order texture blur threshold ratio against face baseline.",
                    ),
                    "export_mask": OpenAIFunctionPropertySchema(
                        type="boolean",
                        description="Whether to export anomaly mask.",
                    ),
                    "output_dir": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Directory to save anomaly mask.",
                    ),
                },
                required=["image_input"],
            ),
            strict=False,
        ),
    )


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _clip(v: float, low: float, high: float) -> float:
    return max(low, min(high, v))


def _bbox_from_points(points: List[Tuple[int, int]], w: int, h: int, pad: float = 0.0) -> Tuple[int, int, int, int]:
    if not points:
        return 0, 0, w, h
    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    x0 = float(np.min(xs))
    x1 = float(np.max(xs))
    y0 = float(np.min(ys))
    y1 = float(np.max(ys))

    pw = (x1 - x0 + 1.0) * pad
    ph = (y1 - y0 + 1.0) * pad

    bx0 = max(0, int(np.floor(x0 - pw)))
    by0 = max(0, int(np.floor(y0 - ph)))
    bx1 = min(w, int(np.ceil(x1 + pw + 1.0)))
    by1 = min(h, int(np.ceil(y1 + ph + 1.0)))

    if bx1 <= bx0:
        bx1 = min(w, bx0 + 1)
    if by1 <= by0:
        by1 = min(h, by0 + 1)

    return bx0, by0, bx1, by1


def _draw_disk(mask: np.ndarray, x: int, y: int, r: int, value: int = 255) -> None:
    h, w = mask.shape
    x0 = max(0, x - r)
    x1 = min(w, x + r + 1)
    y0 = max(0, y - r)
    y1 = min(h, y + r + 1)
    yy, xx = np.indices((y1 - y0, x1 - x0))
    rr2 = (xx + x0 - x) ** 2 + (yy + y0 - y) ** 2
    mask[y0:y1, x0:x1][rr2 <= r * r] = value


def _second_order_energy(gray_patch: np.ndarray) -> float:
    if gray_patch.size == 0:
        return 0.0
    patch = gray_patch.astype(np.float32)
    gx, gy = np.gradient(patch)
    gxx, _ = np.gradient(gx)
    _, gyy = np.gradient(gy)
    lap = np.abs(gxx + gyy)
    return float(np.mean(lap))


class FacialBiologicalAuditor:
    def __init__(
        self,
        default_keypoint_backend: str = "auto",
        default_max_side: int = 768,
        default_reflection_angle_threshold_deg: float = 18.0,
        default_reflection_mag_ratio_threshold: float = 1.8,
        default_texture_blur_ratio_threshold: float = 0.45,
        default_export_mask: bool = True,
        default_output_dir: str = "./outputs/facial_biological_auditor",
    ):
        self.tool_name = "Facial_Biological_Auditor"
        self.default_keypoint_backend = default_keypoint_backend
        self.default_max_side = default_max_side
        self.default_reflection_angle_threshold_deg = default_reflection_angle_threshold_deg
        self.default_reflection_mag_ratio_threshold = default_reflection_mag_ratio_threshold
        self.default_texture_blur_ratio_threshold = default_texture_blur_ratio_threshold
        self.default_export_mask = default_export_mask
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

    def _prepare_image(self, image: Image.Image, max_side: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
        ow, oh = image.size
        scale = 1.0
        if max(ow, oh) > max_side:
            scale = float(max_side) / float(max(ow, oh))
        nw = max(128, int(round(ow * scale)))
        nh = max(128, int(round(oh * scale)))
        rgb_img = image.resize((nw, nh), Image.BILINEAR).convert("RGB")
        rgb = np.asarray(rgb_img, dtype=np.uint8)
        gray = np.asarray(rgb_img.convert("L"), dtype=np.float32)
        return rgb, gray, (ow, oh), (nw, nh)

    def _extract_landmarks_mediapipe(self, rgb: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        if not _HAS_MEDIAPIPE:
            return None
        try:
            h, w, _ = rgb.shape
            face_mesh = mp.solutions.face_mesh.FaceMesh(  # type: ignore
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            result = face_mesh.process(rgb)
            face_mesh.close()
            if not result.multi_face_landmarks:
                return None
            lms = result.multi_face_landmarks[0].landmark
            points: List[Tuple[int, int]] = []
            for lm in lms:
                x = int(round(lm.x * (w - 1)))
                y = int(round(lm.y * (h - 1)))
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                points.append((x, y))
            return points
        except Exception:
            return None

    def _extract_landmarks_insightface(self, rgb: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        if not _HAS_INSIGHTFACE:
            return None
        try:
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])  # type: ignore
            app.prepare(ctx_id=-1)
            faces = app.get(rgb)
            if not faces:
                return None
            face = faces[0]
            kps = getattr(face, "kps", None)
            if kps is None:
                return None
            points: List[Tuple[int, int]] = []
            for p in np.asarray(kps):
                points.append((int(round(float(p[0]))), int(round(float(p[1])))))
            return points
        except Exception:
            return None

    def _extract_landmarks(self, rgb: np.ndarray, backend: str) -> Tuple[Optional[List[Tuple[int, int]]], str]:
        b = backend.strip().lower()
        if b not in {"auto", "mediapipe", "insightface"}:
            b = "auto"

        if b in {"auto", "mediapipe"}:
            pts = self._extract_landmarks_mediapipe(rgb)
            if pts:
                return pts, "mediapipe_468"
            if b == "mediapipe":
                return None, "mediapipe_468"

        if b in {"auto", "insightface"}:
            pts = self._extract_landmarks_insightface(rgb)
            if pts:
                return pts, "insightface"
            return None, "insightface"

        return None, "unknown"

    def _eye_reflection_point(self, rgb: np.ndarray, eye_pts: List[Tuple[int, int]]) -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
        h, w, _ = rgb.shape
        x0, y0, x1, y1 = _bbox_from_points(eye_pts, w=w, h=h, pad=0.25)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None, {"roi": [x0, y0, x1, y1], "reason": "eye_roi_too_small"}

        patch = rgb[y0:y1, x0:x1].astype(np.float32)
        gray = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
        sat = np.mean(patch, axis=2)

        score = gray * (sat / 255.0)
        thr = float(np.percentile(score, 99.6))
        mask = score >= thr

        ys, xs = np.where(mask)
        if ys.size == 0:
            return None, {"roi": [x0, y0, x1, y1], "reason": "no_specular_point"}

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        px = int(round(x0 + cx))
        py = int(round(y0 + cy))
        return (px, py), {"roi": [x0, y0, x1, y1], "threshold": round(thr, 4)}

    def _corneal_reflection_consistency(
        self,
        left_eye_center: Tuple[float, float],
        right_eye_center: Tuple[float, float],
        left_reflection: Optional[Tuple[int, int]],
        right_reflection: Optional[Tuple[int, int]],
        angle_threshold_deg: float,
        mag_ratio_threshold: float,
    ) -> Tuple[Dict[str, Any], bool, List[str], List[Tuple[int, int]]]:
        issues: List[str] = []
        anomaly_pts: List[Tuple[int, int]] = []

        if left_reflection is None or right_reflection is None:
            issues.append("角膜反射点缺失，无法完成双眼光源几何一致性验证")
            return {
                "available": False,
                "angle_diff_deg": None,
                "magnitude_ratio": None,
            }, True, issues, anomaly_pts

        lx, ly = float(left_reflection[0]), float(left_reflection[1])
        rx, ry = float(right_reflection[0]), float(right_reflection[1])
        lcx, lcy = left_eye_center
        rcx, rcy = right_eye_center

        vl = np.array([lx - lcx, ly - lcy], dtype=np.float32)
        vr = np.array([rx - rcx, ry - rcy], dtype=np.float32)

        # Mirror right-eye vector into left-eye coordinate sense.
        vr_m = np.array([-vr[0], vr[1]], dtype=np.float32)

        nl = float(np.linalg.norm(vl)) + 1e-8
        nr = float(np.linalg.norm(vr_m)) + 1e-8

        cosv = float(np.dot(vl, vr_m) / (nl * nr))
        cosv = max(-1.0, min(1.0, cosv))
        angle_diff = float(np.degrees(np.arccos(cosv)))
        mag_ratio = float(max(nl, nr) / (min(nl, nr) + 1e-8))

        is_conflict = False
        if angle_diff > angle_threshold_deg:
            issues.append("反射角物理逻辑冲突")
            is_conflict = True
        if mag_ratio > mag_ratio_threshold:
            issues.append("双眼反射位移比例异常")
            is_conflict = True

        if is_conflict:
            anomaly_pts.extend([(int(lx), int(ly)), (int(rx), int(ry))])

        metrics = {
            "available": True,
            "angle_diff_deg": round(angle_diff, 4),
            "magnitude_ratio": round(mag_ratio, 4),
        }
        return metrics, is_conflict, issues, anomaly_pts

    def _texture_checks(
        self,
        gray: np.ndarray,
        landmarks: List[Tuple[int, int]],
        blur_ratio_threshold: float,
    ) -> Tuple[Dict[str, Any], List[str], List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
        h, w = gray.shape
        issues: List[str] = []
        anomaly_pts: List[Tuple[int, int]] = []
        anomaly_boxes: List[Tuple[int, int, int, int]] = []

        if len(landmarks) < 478:
            return {
                "baseline_energy": None,
                "left_iris_energy": None,
                "right_iris_energy": None,
                "teeth_energy": None,
            }, ["关键点数量不足，无法执行虹膜/牙齿细节审计"], anomaly_pts, anomaly_boxes

        face_box = _bbox_from_points(landmarks, w=w, h=h, pad=0.05)
        fx0, fy0, fx1, fy1 = face_box
        face_patch = gray[fy0:fy1, fx0:fx1]
        base_e = _second_order_energy(face_patch) + 1e-8

        left_iris_idx = [468, 469, 470, 471, 472]
        right_iris_idx = [473, 474, 475, 476, 477]
        mouth_inner_idx = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13]

        left_iris_pts = [landmarks[i] for i in left_iris_idx]
        right_iris_pts = [landmarks[i] for i in right_iris_idx]
        mouth_pts = [landmarks[i] for i in mouth_inner_idx]

        lb = _bbox_from_points(left_iris_pts, w=w, h=h, pad=0.8)
        rb = _bbox_from_points(right_iris_pts, w=w, h=h, pad=0.8)
        tb = _bbox_from_points(mouth_pts, w=w, h=h, pad=0.35)

        lx0, ly0, lx1, ly1 = lb
        rx0, ry0, rx1, ry1 = rb
        tx0, ty0, tx1, ty1 = tb

        le = _second_order_energy(gray[ly0:ly1, lx0:lx1])
        re = _second_order_energy(gray[ry0:ry1, rx0:rx1])
        te = _second_order_energy(gray[ty0:ty1, tx0:tx1])

        left_ratio = le / base_e
        right_ratio = re / base_e
        teeth_ratio = te / base_e
        iris_lr_ratio = max(le, re) / (min(le, re) + 1e-8)

        if left_ratio < blur_ratio_threshold or right_ratio < blur_ratio_threshold:
            issues.append("虹膜纹理二阶梯度异常偏低，疑似生成模糊")
            anomaly_boxes.extend([lb, rb])
            anomaly_pts.extend([(int((lx0 + lx1) / 2), int((ly0 + ly1) / 2)), (int((rx0 + rx1) / 2), int((ry0 + ry1) / 2))])

        if iris_lr_ratio > 1.9:
            issues.append("双眼虹膜纹理清晰度不对称")
            anomaly_boxes.extend([lb, rb])

        if teeth_ratio < blur_ratio_threshold:
            issues.append("牙齿区域纹理梯度异常偏低，疑似深度伪造平滑")
            anomaly_boxes.append(tb)
            anomaly_pts.append((int((tx0 + tx1) / 2), int((ty0 + ty1) / 2)))

        if teeth_ratio > 2.8:
            issues.append("牙齿区域纹理梯度异常偏高，疑似排列/纹理伪影")
            anomaly_boxes.append(tb)

        metrics = {
            "baseline_energy": round(float(base_e), 6),
            "left_iris_energy": round(float(le), 6),
            "right_iris_energy": round(float(re), 6),
            "teeth_energy": round(float(te), 6),
            "left_iris_ratio": round(float(left_ratio), 6),
            "right_iris_ratio": round(float(right_ratio), 6),
            "teeth_ratio": round(float(teeth_ratio), 6),
            "iris_lr_ratio": round(float(iris_lr_ratio), 6),
        }
        return metrics, issues, anomaly_pts, anomaly_boxes

    def __call__(self, args_str: str) -> Dict[str, Any]:
        start_time = time.time()
        log_call_args: Dict[str, Any] = {"raw_args": args_str}

        def _finalize(response_obj: Dict[str, Any]) -> Dict[str, Any]:
            log_tool_use_record(
                tool_key="facial_biological_auditor",
                tool_name=self.tool_name,
                call_args=log_call_args,
                response=response_obj,
            )
            return response_obj

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            return _finalize(self._build_response(False, start_time, "Facial biological audit failed.", error_message="参数必须为 JSON 字符串"))

        image_path = args.get("image_path") or args.get("image_input")
        if not image_path:
            return _finalize(self._build_response(False, start_time, "Facial biological audit failed.", error_message="缺少必填参数 image_path"))
        if not os.path.exists(image_path):
            return _finalize(self._build_response(False, start_time, "Facial biological audit failed.", error_message=f"找不到图像文件: {image_path}"))

        backend = str(args.get("keypoint_backend", self.default_keypoint_backend)).strip().lower()
        max_side = max(256, min(1024, _safe_int(args.get("max_side", self.default_max_side), self.default_max_side)))
        angle_thr = _clip(_safe_float(args.get("reflection_angle_threshold_deg", self.default_reflection_angle_threshold_deg), self.default_reflection_angle_threshold_deg), 5.0, 45.0)
        mag_ratio_thr = _clip(_safe_float(args.get("reflection_mag_ratio_threshold", self.default_reflection_mag_ratio_threshold), self.default_reflection_mag_ratio_threshold), 1.1, 4.0)
        blur_ratio_thr = _clip(_safe_float(args.get("texture_blur_ratio_threshold", self.default_texture_blur_ratio_threshold), self.default_texture_blur_ratio_threshold), 0.2, 0.9)
        export_mask = bool(args.get("export_mask", self.default_export_mask))
        output_dir = str(args.get("output_dir", self.default_output_dir))

        log_call_args = {
            "image_path": image_path,
            "keypoint_backend": backend,
            "max_side": max_side,
            "reflection_angle_threshold_deg": angle_thr,
            "reflection_mag_ratio_threshold": mag_ratio_thr,
            "texture_blur_ratio_threshold": blur_ratio_thr,
            "export_mask": export_mask,
            "output_dir": output_dir,
        }

        try:
            image = Image.open(image_path).convert("RGB")
            rgb, gray, orig_size, proc_size = self._prepare_image(image, max_side=max_side)
        except Exception as exc:
            return _finalize(self._build_response(False, start_time, "Facial biological audit failed.", error_message=f"图像读取失败: {exc}"))

        landmarks, kp_model = self._extract_landmarks(rgb, backend=backend)
        if not landmarks:
            return _finalize(
                self._build_response(
                    False,
                    start_time,
                    "Facial biological audit failed.",
                    error_message="未检测到可用面部关键点（MediaPipe/InsightFace 不可用或未检测到人脸）。",
                )
            )

        h, w = gray.shape
        anomaly_mask = np.zeros((h, w), dtype=np.uint8)
        issues_all: List[str] = []

        # Eye centers based on MediaPipe key points when available.
        if len(landmarks) >= 468:
            left_eye_idx = [33, 133, 159, 145]
            right_eye_idx = [362, 263, 386, 374]
        else:
            # For sparse sets fallback: split by x around center.
            cx = int(np.mean([p[0] for p in landmarks]))
            left_eye_idx = [i for i, p in enumerate(landmarks) if p[0] < cx][:4]
            right_eye_idx = [i for i, p in enumerate(landmarks) if p[0] >= cx][:4]

        left_eye_pts = [landmarks[i] for i in left_eye_idx if i < len(landmarks)]
        right_eye_pts = [landmarks[i] for i in right_eye_idx if i < len(landmarks)]

        if not left_eye_pts or not right_eye_pts:
            return _finalize(
                self._build_response(
                    False,
                    start_time,
                    "Facial biological audit failed.",
                    error_message="眼部关键点不足，无法进行生物一致性审计。",
                )
            )

        left_eye_center = (float(np.mean([p[0] for p in left_eye_pts])), float(np.mean([p[1] for p in left_eye_pts])))
        right_eye_center = (float(np.mean([p[0] for p in right_eye_pts])), float(np.mean([p[1] for p in right_eye_pts])))

        l_ref, l_ref_meta = self._eye_reflection_point(rgb, left_eye_pts)
        r_ref, r_ref_meta = self._eye_reflection_point(rgb, right_eye_pts)

        refl_metrics, refl_conflict, refl_issues, refl_pts = self._corneal_reflection_consistency(
            left_eye_center=left_eye_center,
            right_eye_center=right_eye_center,
            left_reflection=l_ref,
            right_reflection=r_ref,
            angle_threshold_deg=angle_thr,
            mag_ratio_threshold=mag_ratio_thr,
        )
        issues_all.extend(refl_issues)
        for p in refl_pts:
            _draw_disk(anomaly_mask, p[0], p[1], r=5, value=255)

        tex_metrics, tex_issues, tex_pts, tex_boxes = self._texture_checks(
            gray=gray,
            landmarks=landmarks,
            blur_ratio_threshold=blur_ratio_thr,
        )
        issues_all.extend(tex_issues)
        for p in tex_pts:
            _draw_disk(anomaly_mask, p[0], p[1], r=6, value=255)
        for (x0, y0, x1, y1) in tex_boxes:
            anomaly_mask[y0:y1, x0:x1] = np.maximum(anomaly_mask[y0:y1, x0:x1], 180)

        # Additional points for reflection centers.
        _draw_disk(anomaly_mask, int(round(left_eye_center[0])), int(round(left_eye_center[1])), r=3, value=140)
        _draw_disk(anomaly_mask, int(round(right_eye_center[0])), int(round(right_eye_center[1])), r=3, value=140)

        if l_ref is not None:
            _draw_disk(anomaly_mask, l_ref[0], l_ref[1], r=3, value=220)
        if r_ref is not None:
            _draw_disk(anomaly_mask, r_ref[0], r_ref[1], r=3, value=220)

        # Convert anomaly coordinates back to original image scale.
        ow, oh = orig_size
        pw, ph = proc_size
        sx = float(ow) / float(pw)
        sy = float(oh) / float(ph)

        ys, xs = np.where(anomaly_mask >= 180)
        anomaly_points_proc = [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]
        anomaly_points_global = [
            {"x": int(round(x * sx)), "y": int(round(y * sy))}
            for (x, y) in anomaly_points_proc[:2000]
        ]

        mask_path = ""
        if export_mask:
            os.makedirs(output_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(output_dir, f"{stem}_facial_bio_anomaly_mask.png")
            Image.fromarray(anomaly_mask, mode="L").save(mask_path)

        if not issues_all:
            issues_all.append("未发现显著生物学违和点")

        # Deduplicate issues while preserving order.
        dedup_issues: List[str] = []
        seen = set()
        for x in issues_all:
            if x not in seen:
                seen.add(x)
                dedup_issues.append(x)

        score = 0.0
        if refl_conflict:
            score += 0.45
        score += min(0.35, 0.15 * sum([1 for s in dedup_issues if "虹膜" in s or "牙齿" in s]))
        score += min(0.20, len(anomaly_points_proc) / float(max(1, pw * ph)) * 40.0)
        anomaly_point_ratio = len(anomaly_points_proc) / float(max(1, pw * ph))
        evidence_scores = {
            "reflection_conflict": bool(refl_conflict),
            "issue_count": int(len(dedup_issues)),
            "anomaly_point_ratio": round(float(anomaly_point_ratio), 8),
            "signal_strength": round(float(max(0.0, min(1.0, score))), 6),
        }

        result = {
            "forensics_report": {
                "output_mode": "evidence_only",
                "method": "Facial biological consistency audit",
                "summary": "已输出面部生物一致性观测与异常点信息；不包含最终判定结论。",
            },
            "evidence_scores": evidence_scores,
            "biological_inconsistencies": dedup_issues,
            "reflection_analysis": {
                "metrics": refl_metrics,
                "left_reflection": {"x": l_ref[0], "y": l_ref[1]} if l_ref else None,
                "right_reflection": {"x": r_ref[0], "y": r_ref[1]} if r_ref else None,
                "left_meta": l_ref_meta,
                "right_meta": r_ref_meta,
            },
            "semantic_texture_analysis": tex_metrics,
            "anomaly_mask": {
                "processed_size": {"width": int(pw), "height": int(ph)},
                "original_size": {"width": int(ow), "height": int(oh)},
                "anomaly_points_global": anomaly_points_global,
                "anomaly_point_count": len(anomaly_points_global),
                "mask_path": mask_path,
            },
            "landmark_meta": {
                "backend": kp_model,
                "landmark_count": len(landmarks),
            },
        }

        return _finalize(self._build_response(True, start_time, "Facial biological audit completed.", result=result))


class FacialBiologicalAuditorTool(BaseTool):
    """verl-native wrapper aligned with BaseTool lifecycle."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema or _default_schema())
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self._default_keypoint_backend = str(config.get("default_keypoint_backend", "auto"))
        self._default_max_side = int(config.get("default_max_side", 768))
        self._default_reflection_angle_threshold_deg = float(config.get("default_reflection_angle_threshold_deg", 18.0))
        self._default_reflection_mag_ratio_threshold = float(config.get("default_reflection_mag_ratio_threshold", 1.8))
        self._default_texture_blur_ratio_threshold = float(config.get("default_texture_blur_ratio_threshold", 0.45))
        self._default_export_mask = bool(config.get("default_export_mask", True))
        self._default_output_dir = str(config.get("output_dir", "./outputs/facial_biological_auditor"))

        self._tool_impl = FacialBiologicalAuditor(
            default_keypoint_backend=self._default_keypoint_backend,
            default_max_side=self._default_max_side,
            default_reflection_angle_threshold_deg=self._default_reflection_angle_threshold_deg,
            default_reflection_mag_ratio_threshold=self._default_reflection_mag_ratio_threshold,
            default_texture_blur_ratio_threshold=self._default_texture_blur_ratio_threshold,
            default_export_mask=self._default_export_mask,
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
            call_args.setdefault("keypoint_backend", self._default_keypoint_backend)
            call_args.setdefault("max_side", self._default_max_side)
            call_args.setdefault("reflection_angle_threshold_deg", self._default_reflection_angle_threshold_deg)
            call_args.setdefault("reflection_mag_ratio_threshold", self._default_reflection_mag_ratio_threshold)
            call_args.setdefault("texture_blur_ratio_threshold", self._default_texture_blur_ratio_threshold)
            call_args.setdefault("export_mask", self._default_export_mask)
            call_args.setdefault("output_dir", self._default_output_dir)

            result = self._tool_impl(json.dumps(call_args, ensure_ascii=False))
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["response"] = result
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as exc:
            logger.exception("FacialBiologicalAuditorTool execution failed.")
            error_result = {
                "success": False,
                "status": "error",
                "description": "Facial biological audit failed.",
                "result": {},
                "error_message": f"Tool runtime error: {exc}",
            }
            return ToolResponse(text=json.dumps(error_result, ensure_ascii=False)), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
