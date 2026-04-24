# IDF_T 工具说明文档

本文档用于说明 work/IDF_T/tools 目录下当前可用工具，并定义新增工具时的同步更新清单。

## 目录范围

当前结构：
- TF/: 传统伪造检测工具
- AM/: AI 操纵相关检测工具
- GE/: 通用取证与流程支撑工具
- verl_tool_config.yaml: verl 运行时工具注册文件
- __init__.py: 包导出入口

## 工具列表（AM）

### 1) Spectrum_Grid_Analyzer
- 文件：AM/Spectrum_Grid_Analyzer.py
- 类名：SpectrumGridAnalyzerTool
- 目标：在频域识别 AI 操纵常见的网格峰值与能量异常，并定位其空间伪影位置。
- 核心方法：
  - 2D 离散傅里叶变换（DFT）与频谱分析
  - 自动识别偏离中心轴的异常亮点（Grid Peaks）
  - 计算径向高频能量衰减曲线并检测突变空洞（Energy Hole）
  - 将异常频率掩码逆变换到空间域，输出伪影位置图
- 主要输入：
  - image_input / image_path
  - peak_sigma（默认 3.0）
  - center_exclusion_ratio（默认 0.08）
  - energy_hole_sensitivity（默认 2.2）
  - max_side（默认 768）
  - export_map（默认 true）
  - output_dir
- 主要输出：
  - forensics_report.output_mode|method
  - evidence_scores（证据强度分解，不含最终判定）
  - spectrum_statistics（peak_count、energy_hole_count 等）
  - top_grid_peaks（频域异常点列表）
  - artifacts.spectrum_logmag_path
  - artifacts.grid_peak_mask_path
  - artifacts.artifact_spatial_map_path

### 2) Facial_Biological_Auditor
- 文件：AM/Facial_Biological_Auditor.py
- 类名：FacialBiologicalAuditorTool
- 目标：面向深度伪造人脸，审计关键生物学与物理一致性（角膜反射、虹膜纹理、牙齿细节）。
- 核心方法：
  - 关键点对齐：优先 MediaPipe（468 点，含虹膜点），可回退 InsightFace
  - 物理一致性：双眼角膜反射点几何一致性检查（反射角与位移比例）
  - 语义细节：牙齿区域与虹膜区域二阶梯度（Laplacian-like）异常检测
  - 生成异常掩码并映射异常点坐标
- 主要输入：
  - image_input / image_path
  - keypoint_backend（auto/mediapipe/insightface）
  - max_side（默认 768）
  - reflection_angle_threshold_deg（默认 18）
  - reflection_mag_ratio_threshold（默认 1.8）
  - texture_blur_ratio_threshold（默认 0.45）
  - export_mask（默认 true）
  - output_dir
- 主要输出：
  - forensics_report.output_mode|method
  - evidence_scores（证据强度分解，不含最终判定）
  - biological_inconsistencies（例如“反射角物理逻辑冲突”）
  - reflection_analysis（双眼反射几何指标）
  - semantic_texture_analysis（牙齿/虹膜二阶梯度统计）
  - anomaly_mask.anomaly_points_global
  - anomaly_mask.mask_path

## 工具列表（GE）

### 1) ROI_Extractor
- 文件：GE/ROI_Extractor.py
- 类名：ROIExtractorTool
- 目标：按坐标快速裁剪子图，并支持局部掩码回溯到原图全局坐标系。
- 核心方法：
  - 支持绝对坐标/相对坐标 ROI 裁剪
  - 支持缩放坐标系到原图坐标系映射（coordinate_space=scaled）
  - 将局部检测掩码映射回全局坐标，输出全局 bbox 与映射变换参数
- 主要输入：
  - image_input / image_path
  - roi 或 x,y,w,h
  - coord_type（abs/relative）
  - coordinate_space（original/scaled）
  - source_width/source_height（scaled 坐标时使用）
  - mask_input（可选）
  - local_ref_width/local_ref_height（可选）
  - mask_threshold（默认 0.5）
  - export_crop（默认 true）
  - export_global_mask（默认 true）
  - output_dir
- 主要输出：
  - roi_result.global_roi_box 与相对坐标
  - mask_backtracking.global_mask_bbox
  - mask_backtracking.mapping_transform
  - artifacts.roi_crop_path
  - artifacts.global_mask_path

### 2) Image_Resizer
- 文件：GE/Image_Resizer.py
- 类名：ImageResizerTool
- 目标：执行像素级图像缩放，并记录缩放因子与重采样元数据。
- 核心方法：
  - 严格插值算法控制（nearest/bilinear/bicubic/lanczos）
  - 支持按目标尺寸或缩放因子 s 调整分辨率
  - 输出缩放因子 sx、sy、s，并给出对取证敏感度的影响提示
- 取证特化：
  - 放大查看可通过 nearest 保持像素值不被平滑
  - 缩小适配模型时记录 s，便于解释 SRM_Filter 等工具敏感度变化
- 主要输入：
  - image_input / image_path
  - target_width/target_height 或 scale_factor
  - keep_aspect_ratio
  - interpolation
  - prefer_forensics_mode（auto/pixel_inspection/model_adaptation）
  - output_format（PNG/JPEG/WEBP）
  - jpeg_quality（JPEG 时有效）
  - output_dir
- 主要输出：
  - resize_report.original_size/target_size
  - resize_report.scale_factor（sx/sy/s/is_uniform）
  - resize_report.interpolation
  - resize_report.resampling_applied
  - resize_report.sensitivity_hint
  - artifacts.resized_image_path

### 3) Visual_Enhancer
- 文件：GE/Visual_Enhancer.py
- 类名：VisualEnhancerTool
- 目标：通过局部自适应亮度/对比度增强，提升极暗或极亮区域内隐藏伪影的可见性。
- 核心方法：
  - CLAHE 风格局部对比度增强（clip_limit + tile_grid_size）
  - 可选亮度、对比度、gamma 后处理
  - 输出伪影可见性图（增强前后亮度差）
- 取证特化：
  - 对可能被光照或过曝/欠曝掩盖的接缝区域进行显著性提升
  - 为 Agent 提供可用于二次取证的增强图与可见性图
- 主要输入：
  - image_input / image_path
  - method（当前支持 clahe）
  - clip_limit（默认 2.0）
  - tile_grid_size（默认 8）
  - brightness_gain / contrast_gain / gamma
  - output_format（PNG/JPEG/WEBP）
  - jpeg_quality（JPEG 时有效）
  - output_dir
- 主要输出：
  - enhancement_report（动态范围变化、暗区/亮区比例变化、取证提示）
  - artifacts.enhanced_image_path
  - artifacts.artifact_visibility_map_path

## 工具列表（TF）

### 1) SRM_Filter
- 文件：TF/SRM_Filter.py
- 类名：SRMFilterTool
- 目标：通过 SRM 高阶残差滤波器组抑制图像语义信息，放大亚像素统计噪声痕迹。
- 核心方法：
  - 多核高阶残差滤波
  - 残差截断
  - 聚合可疑热力图
- 主要输入：
  - image_input / image_path
  - truncate_threshold（默认 3.0）
  - export_map（默认 true）
  - output_dir
- 主要输出：
  - forensics_report.output_mode|method
  - evidence_scores（证据强度分解，不含最终判定）
  - residual_statistics
  - artifacts.srm_heatmap_path

### 2) Noise_Analyzer
- 文件：TF/Noise_Analyzer.py
- 类名：NoiseAnalyzerTool
- 目标：通过局部噪声方差不一致性检测拼接（Splicing）区域。
- 核心方法：
  - Haar 小波分解（HH 高频子带）
  - 局部方差估计
  - 多层聚合噪声方差热力图
- 主要输入：
  - image_input / image_path
  - window_size（默认 9）
  - levels（默认 2）
  - export_map（默认 true）
  - output_dir
- 主要输出：
  - forensics_report.output_mode|method
  - evidence_scores（证据强度分解，不含最终判定）
  - noise_statistics
  - artifacts.noise_variance_heatmap_path

### 3) PatchDense_Scanner
- 文件：TF/PatchDense_Scanner.py
- 类名：PatchDenseScannerTool
- 目标：通过密集偏移场一致性检测 copy-move 剪切痕迹。
- 核心方法：
  - 类 PatchMatch 的密集最近邻偏移搜索
  - Thresholded Offset Consistency（TOC）
  - 由一致性、相似度与位移门控融合得到可疑热力图
- 主要输入：
  - image_input / image_path
  - patch_size（默认 7）
  - iterations（默认 4）
  - consistency_threshold（默认 0.6）
  - max_side（默认 384）
  - export_map（默认 true）
  - output_dir
- 主要输出：
  - forensics_report.output_mode|method
  - evidence_scores（证据强度分解，不含最终判定）
  - offset_statistics
  - artifacts.copymove_heatmap_path

### 4) CFA_Validator
- 文件：TF/CFA_Validator.py
- 类名：CFAValidatorTool
- 目标：通过拜耳 CFA 相位一致性验证检测 inpainting 填充痕迹。
- 核心方法：
  - 4 种 Bayer 相位假设
  - 伪马赛克重建误差分析
  - 像素级不连续性概率估计
- 主要输入：
  - image_input / image_path
  - window_size（默认 7）
  - prob_threshold（默认 0.65）
  - max_side（默认 512）
  - export_map（默认 true）
  - output_dir
- 主要输出：
  - forensics_report.output_mode|method
  - evidence_scores（证据强度分解，不含最终判定）
  - cfa_statistics
  - artifacts.cfa_discontinuity_probability_map_path

## 统一响应协议

所有工具应返回以下统一字段：
- success: bool
- status: success|error
- description: 简要说明
- result: 工具结构化结果
- error_message: 空或错误信息
- metadata.tool_name
- metadata.execution_time_ms

## 注册与导出要求

每次新增工具时，必须同时完成以下更新：
1. 在对应子目录新增工具文件（例如传统工具放在 TF/，通用工具放在 GE/）。
2. 在对应子目录的 __init__.py 导出工具类（如 TF/__init__.py 或 GE/__init__.py）。
3. 在 tools/__init__.py 导出工具类。
4. 在 verl_tool_config.yaml 中注册 class_path 与默认配置。
5. 更新本文件 TOOLS.md，补充：
   - 工具目标
   - 算法摘要
   - 输入参数
   - 输出字段

## 新增工具检查清单

新增工具时请确认：
- 命名规范：
  - 文件名：Name_With_Underscore.py
  - 运行时类名：NameWithCamelTool
- Schema 规范：
  - 包含 image_input 与 image_path 别名
  - 明确默认值和取值范围
- 运行时包装：
  - 实现 create/execute/release
  - 保持统一响应协议
- 可视化产物：
  - 由 export_map 控制是否导出
  - 路径写入 result.artifacts.*
- 配置注册：
  - 在 verl_tool_config.yaml 中新增一条配置
- 文档更新：
  - 在本文件追加对应工具说明

## 维护规则

只要发生以下任一变更，必须在同一次提交中同步更新本文件：
- 新增工具
- 删除工具
- 输入参数变更
- 输出字段或结构变更
