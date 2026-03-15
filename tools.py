"""
tools.py — 工具注册表与医学影像分析函数
==========================================

本模块实现:
1. ToolRegistry: 通过字典将 Tool ID 映射到具体的 Python 函数。
2. 通用医学图像加载器: 支持 .zip (DICOM), .nii.gz, .mhd/.raw 格式。
3. 肺部分割: 使用 lungmask 库进行双肺分割。
4. 肺结节检测: 使用 MONAI RetinaNet 3D 预训练模型进行结节检测。

技术栈:
    - SimpleITK: 医学图像 I/O 与格式转换
    - lungmask: 基于 U-Net 的肺实质分割
    - MONAI: 3D 目标检测 (RetinaNet)
    - PyTorch: 深度学习推理
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# ---- 医学影像库 (延迟导入以加速启动) ----
# SimpleITK, nibabel, torch, monai, lungmask 在函数内部按需导入

from schemas import DetectionOutput, NoduleFinding, SegmentationOutput


# ============================================================
# 工具注册表
# ============================================================

class ToolRegistry:
    """将整数 Tool ID 映射到可调用的 Python 函数。

    使用方式:
        registry = ToolRegistry()
        registry.register(100, detect_nodules)
        result = registry.call(100, image_path="scan.nii.gz")
    """

    def __init__(self) -> None:
        self._tools: Dict[int, Callable[..., Any]] = {}

    def register(self, tool_id: int, fn: Callable[..., Any]) -> None:
        """注册一个工具函数到指定 ID。"""
        self._tools[tool_id] = fn

    def call(self, tool_id: int, **kwargs: Any) -> Any:
        """根据 Tool ID 调用对应函数并返回结果。

        Raises:
            KeyError: 如果 tool_id 未注册。
        """
        if tool_id not in self._tools:
            raise KeyError(
                f"Tool ID {tool_id} 未注册。已注册的工具: {list(self._tools.keys())}"
            )
        return self._tools[tool_id](**kwargs)

    @property
    def registered_ids(self) -> list[int]:
        """返回所有已注册的 Tool ID 列表。"""
        return list(self._tools.keys())


# ============================================================
# 通用医学图像加载器
# ============================================================

def load_medical_image(file_path: str) -> "SimpleITK.Image":
    """通用医学图像加载函数，支持多种格式。

    支持的格式:
        - .zip  → 解压后使用 ImageSeriesReader 读取 DICOM 序列
        - .nii / .nii.gz → 直接读取 NIfTI 文件
        - .mhd  → 直接读取 MetaImage 文件 (自动关联 .raw)

    返回:
        SimpleITK.Image 对象，保留完整的 Spacing / Origin / Direction 信息。

    注意:
        医学图像的物理坐标系 (World Coordinate):
        - Origin: 图像左上角第一个体素的物理坐标 (mm)
        - Spacing: 每个体素在 X/Y/Z 方向的物理尺寸 (mm)
        - Direction: 3x3 方向余弦矩阵，定义体素轴与物理轴的映射关系
        这些参数对于将体素坐标转换为世界坐标至关重要。
    """
    import SimpleITK as sitk

    file_path = str(file_path)
    lower = file_path.lower()

    if lower.endswith(".zip"):
        # ---------- ZIP (DICOM 序列) ----------
        # 解压到临时目录，扫描 DICOM 文件并读取为 3D 体数据
        extract_dir = tempfile.mkdtemp(prefix="medclaw_dicom_")
        print(f"[ImageLoader] 解压 ZIP 到: {extract_dir}")

        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(extract_dir)

        # 递归查找包含 DICOM 文件的目录
        dicom_dir = _find_dicom_directory(extract_dir)
        if dicom_dir is None:
            raise FileNotFoundError(
                f"ZIP 文件中未找到有效的 DICOM 序列: {file_path}"
            )

        # 使用 ImageSeriesReader 读取整个 DICOM 序列
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
        if not series_ids:
            raise FileNotFoundError(f"目录中无 DICOM Series: {dicom_dir}")

        # 取第一个 Series（通常是主扫描）
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
        reader.SetFileNames(dicom_files)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        image = reader.Execute()
        print(
            f"[ImageLoader] DICOM 加载完成: "
            f"Size={image.GetSize()}, "
            f"Spacing={image.GetSpacing()}, "
            f"Origin={image.GetOrigin()}"
        )
        return image

    elif lower.endswith((".nii", ".nii.gz")):
        # ---------- NIfTI ----------
        image = sitk.ReadImage(file_path)
        print(
            f"[ImageLoader] NIfTI 加载完成: "
            f"Size={image.GetSize()}, "
            f"Spacing={image.GetSpacing()}"
        )
        return image

    elif lower.endswith(".mhd"):
        # ---------- MetaImage (.mhd + .raw) ----------
        image = sitk.ReadImage(file_path)
        print(
            f"[ImageLoader] MHD 加载完成: "
            f"Size={image.GetSize()}, "
            f"Spacing={image.GetSpacing()}"
        )
        return image

    else:
        raise ValueError(f"不支持的文件格式: {file_path}")


def _find_dicom_directory(root_dir: str) -> Optional[str]:
    """递归查找包含 DICOM 文件的子目录。"""
    import SimpleITK as sitk

    for dirpath, _, filenames in os.walk(root_dir):
        # 检查目录中是否有 .dcm 文件或无扩展名的 DICOM 文件
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dirpath)
        if series_ids:
            return dirpath
    return None


# ============================================================
# Step 1: 肺部分割 — 使用 lungmask 库
# ============================================================

def segment_lungs(image_path: str, **kwargs: Any) -> dict:
    """对胸部 CT 进行双肺分割。

    技术选型: lungmask (https://github.com/JoHof/lungmask)
    - 使用预训练的 U-Net 模型 (R231 为默认模型)
    - 输入: 3D CT 体数据
    - 输出: 双肺 mask (左肺=1, 右肺=2)

    Args:
        image_path: 医学影像文件路径。

    Returns:
        包含分割结果的字典 (SegmentationOutput 格式)。
    """
    import SimpleITK as sitk

    print(f"[Tool 101] 🫁 开始肺部分割: {image_path}")

    # 1. 加载图像
    sitk_image = load_medical_image(image_path)

    # 2. 创建输出目录
    output_dir = tempfile.mkdtemp(prefix="medclaw_seg_")
    mask_path = os.path.join(output_dir, "lung_mask.nii.gz")

    # 3. 执行肺分割
    try:
        from lungmask import LMInferer

        # R231 是 lungmask 的默认全肺分割模型
        inferer = LMInferer(modelname="R231")

        # lungmask 接受 SimpleITK Image 对象
        # 返回 numpy 数组: 0=背景, 1=左肺, 2=右肺
        mask_array = inferer.apply(sitk_image)

        # 4. 将 mask 保存为 NIfTI 文件，保持与原图一致的空间信息
        # 注意: 必须保留 Origin / Spacing / Direction，
        # 这样后续结节检测可以正确映射到世界坐标
        mask_sitk = sitk.GetImageFromArray(mask_array)
        mask_sitk.CopyInformation(sitk_image)  # 复制空间元数据
        sitk.WriteImage(mask_sitk, mask_path)

        # 5. 计算肺体积 (体素数 × 体素物理体积)
        spacing = sitk_image.GetSpacing()  # (sx, sy, sz) in mm
        voxel_volume_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0  # mm³ → mL

        left_volume = float(np.sum(mask_array == 1)) * voxel_volume_ml
        right_volume = float(np.sum(mask_array == 2)) * voxel_volume_ml

        print(
            f"[Tool 101] ✅ 肺分割完成 — "
            f"左肺: {left_volume:.0f}mL, 右肺: {right_volume:.0f}mL"
        )

        result = SegmentationOutput(
            tool_id=101,
            tool_name="肺部分割",
            image_path=image_path,
            mask_path=mask_path,
            left_lung_volume_ml=round(left_volume, 1),
            right_lung_volume_ml=round(right_volume, 1),
            status="success",
        )
        return result.model_dump()

    except ImportError:
        print("[Tool 101] ⚠️ lungmask 未安装，使用 Mock 分割结果")
        return _mock_segment_lungs(image_path, mask_path)

    except Exception as e:
        print(f"[Tool 101] ❌ 肺分割失败: {e}")
        return _mock_segment_lungs(image_path, mask_path)


def _mock_segment_lungs(image_path: str, mask_path: str) -> dict:
    """当 lungmask 不可用时的 Mock 兜底。"""
    time.sleep(0.5)
    result = SegmentationOutput(
        tool_id=101,
        tool_name="肺部分割（Mock）",
        image_path=image_path,
        mask_path=mask_path,
        left_lung_volume_ml=2450.0,
        right_lung_volume_ml=2680.0,
        status="mock",
    )
    return result.model_dump()


# ============================================================
# Step 2: 肺结节检测 — 使用 MONAI RetinaNet 3D
# ============================================================

# MONAI Bundle 路径 (相对于项目根目录)
BUNDLE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "monai_bundles",
    "lung_nodule_ct_detection",
)


def detect_nodules(image_path: str, **kwargs: Any) -> dict:
    """使用 MONAI RetinaNet 3D 检测肺结节。

    技术选型: MONAI Model Zoo — lung_nodule_ct_detection Bundle
    - 基于 RetinaNet 3D + ResNet50-FPN 骨干网络
    - 预训练于 LUNA16 数据集
    - 使用滑动窗口推理 (Sliding Window Inference) 处理全分辨率 CT

    推理流程:
        1. 加载 CT 图像 (SimpleITK → numpy)
        2. 预处理: 通道扩展 → RAS 定向 → 重采样 → HU 窗值归一化
        3. RetinaNet 推理 (滑动窗口)
        4. 后处理: NMS → 体素坐标转世界坐标 → 格式化输出

    Voxel to World Coordinate 转换说明:
        世界坐标 = Direction × (VoxelIndex × Spacing) + Origin
        其中:
        - VoxelIndex: 体素在图像矩阵中的位置 [i, j, k]
        - Spacing: 体素间距 [sx, sy, sz] (mm)
        - Direction: 3×3 方向余弦矩阵
        - Origin: 图像原点的物理坐标 [ox, oy, oz] (mm)
        MONAI 的 AffineBoxToWorldCoordinated 自动完成此转换。

    Args:
        image_path: 医学影像文件路径。
        **kwargs: 可选参数，如 mask_path (肺 mask 路径用于过滤假阳性)。

    Returns:
        包含检测结果的字典 (DetectionOutput 格式)。
    """
    mask_path = kwargs.get("mask_path")
    print(f"[Tool 100] 🔍 开始结节检测: {image_path}")

    try:
        return _run_monai_detection(image_path, mask_path)
    except ImportError as e:
        print(f"[Tool 100] ⚠️ MONAI 依赖缺失 ({e})，使用 Mock 检测结果")
        return _mock_detect_nodules(image_path, mask_path)
    except Exception as e:
        print(f"[Tool 100] ❌ MONAI 检测失败: {e}")
        return _mock_detect_nodules(image_path, mask_path)


def _run_monai_detection(image_path: str, mask_path: Optional[str]) -> dict:
    """执行真实的 MONAI RetinaNet 3D 结节检测推理。"""
    import torch
    import monai
    from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
    from monai.apps.detection.networks.retinanet_network import (
        RetinaNet,
        resnet_fpn_feature_extractor,
    )
    from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
    from monai.networks.nets.resnet import resnet50
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
    )
    from monai.apps.detection.transforms.dictionary import (
        AffineBoxToWorldCoordinated,
        ClipBoxToImaged,
        ConvertBoxModed,
    )
    from monai.data import Dataset, DataLoader
    import SimpleITK as sitk

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Tool 100] 使用设备: {device}")

    # ---------- 1. 构建网络 ----------
    # 与 inference.json 配置保持一致
    backbone = resnet50(
        spatial_dims=3,
        n_input_channels=1,
        conv1_t_stride=[2, 2, 1],
        conv1_t_size=[7, 7, 7],
    )
    feature_extractor = resnet_fpn_feature_extractor(
        backbone, 3, False, [1, 2], None
    )
    network = RetinaNet(
        spatial_dims=3,
        num_classes=1,
        num_anchors=3,
        feature_extractor=feature_extractor,
        size_divisible=[16, 16, 8],
    ).to(device)

    # ---------- 2. 加载预训练权重 ----------
    model_path = os.path.join(BUNDLE_ROOT, "models", "model.pt")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # MONAI bundle 的 checkpoint 可能是 dict 或直接 state_dict
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            network.load_state_dict(checkpoint["model"])
        else:
            network.load_state_dict(checkpoint)
        print(f"[Tool 100] ✅ 加载预训练权重: {model_path}")
    else:
        print(f"[Tool 100] ⚠️ 未找到预训练权重: {model_path}")

    network.eval()

    # ---------- 3. 构建检测器 ----------
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[1, 2, 4],
        base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]],
    )
    detector = RetinaNetDetector(
        network=network,
        anchor_generator=anchor_generator,
        debug=False,
        spatial_dims=3,
        num_classes=1,
        size_divisible=[16, 16, 8],
    )
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(
        score_thresh=0.02,
        topk_candidates_per_level=1000,
        nms_thresh=0.22,
        detections_per_img=300,
    )

    # 滑动窗口推理参数
    infer_patch_size = [512, 512, 192]
    detector.set_sliding_window_inferer(
        roi_size=infer_patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        device="cpu",
    )

    # ---------- 4. 预处理 ----------
    # 加载后需要保存原始图像路径以获取仿射矩阵用于坐标转换
    preprocessing = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=[0.703125, 0.703125, 1.25]),
        ScaleIntensityRanged(
            keys="image",
            a_min=-1024.0,
            a_max=300.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys="image"),
    ])

    # ---------- 5. 加载数据 ----------
    # 先确保图像文件可被 MONAI 读取
    # 如果是 .zip，先转换为 NIfTI
    actual_image_path = _prepare_image_for_monai(image_path)

    data = [{"image": actual_image_path}]
    dataset = Dataset(data=data, transform=preprocessing)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=monai.data.utils.no_collation,
    )

    # ---------- 6. 推理 ----------
    print("[Tool 100] 🔄 开始推理 (可能需要几分钟)...")
    results = []

    with torch.no_grad():
        for batch_data in dataloader:
            batch_data_item = batch_data[0]
            image_tensor = batch_data_item["image"].to(device)

            # 使用检测器的推理方法
            # detector 在 eval 模式下返回 [boxes, labels, scores]
            use_inferer = (
                image_tensor.shape[-3] > infer_patch_size[0]
                or image_tensor.shape[-2] > infer_patch_size[1]
                or image_tensor.shape[-1] > infer_patch_size[2]
            )
            outputs = detector(
                [image_tensor],
                use_inferer=use_inferer,
            )

            # outputs 是一个列表，每个元素是 dict: {box, label, label_scores}
            if outputs and len(outputs) > 0:
                pred = outputs[0]
                boxes = pred.get("box", torch.tensor([]))
                scores = pred.get("label_scores", torch.tensor([]))

                # 后处理: 将体素坐标下的检测框转换为世界坐标
                # 使用 MONAI 的 AffineBoxToWorldCoordinated 实现自动转换
                if len(boxes) > 0:
                    post_data = {
                        "box": boxes.cpu(),
                        "label": pred.get("label", torch.tensor([])).cpu(),
                        "image": batch_data_item["image"],
                    }

                    # 裁剪到图像范围
                    clip_transform = ClipBoxToImaged(
                        box_keys="box",
                        label_keys="label",
                        box_ref_image_keys="image",
                        remove_empty=True,
                    )
                    post_data = clip_transform(post_data)

                    # 体素坐标 → 世界坐标 (关键步骤)
                    # 利用图像的仿射矩阵自动完成变换:
                    # world_coord = affine @ voxel_coord
                    affine_transform = AffineBoxToWorldCoordinated(
                        box_keys="box",
                        box_ref_image_keys="image",
                        affine_lps_to_ras=True,
                    )
                    post_data = affine_transform(post_data)

                    # 转换框格式: xyzxyz → cccwhd (中心点 + 宽高深)
                    convert_transform = ConvertBoxModed(
                        box_keys="box",
                        src_mode="xyzxyz",
                        dst_mode="cccwhd",
                    )
                    post_data = convert_transform(post_data)

                    world_boxes = post_data["box"]  # [N, 6] 中心xyz + 宽高深

                    # 过滤低置信度 + 提取分数
                    for i in range(len(world_boxes)):
                        score = float(scores[i]) if i < len(scores) else 0.0
                        if score < 0.1:  # 置信度过滤
                            continue

                        box = world_boxes[i]
                        cx, cy, cz = float(box[0]), float(box[1]), float(box[2])
                        w, h, d = float(box[3]), float(box[4]), float(box[5])
                        # 估算等效球直径: 取宽高深的几何平均值
                        diameter = (w * h * d) ** (1.0 / 3.0)

                        results.append({
                            "center": [cx, cy, cz],
                            "diameter_mm": diameter,
                            "confidence": score,
                        })

    # ---------- 7. 可选: 使用肺 mask 过滤假阳性 ----------
    if mask_path and os.path.exists(mask_path):
        results = _filter_by_lung_mask(results, mask_path, actual_image_path)

    # ---------- 8. 格式化输出 ----------
    # 按置信度降序排序
    results.sort(key=lambda x: x["confidence"], reverse=True)

    findings = []
    for idx, r in enumerate(results):
        findings.append(NoduleFinding(
            nodule_id=idx + 1,
            coordinates_mm=r["center"],
            diameter_mm=round(r["diameter_mm"], 2),
            confidence_score=round(r["confidence"], 4),
        ))

    summary_parts = [f"共检测到 {len(findings)} 枚肺结节。"]
    for f in findings[:5]:  # 报告前 5 个
        summary_parts.append(
            f"结节 #{f.nodule_id}（"
            f"坐标 [{f.coordinates_mm[0]:.1f}, {f.coordinates_mm[1]:.1f}, {f.coordinates_mm[2]:.1f}], "
            f"直径 {f.diameter_mm:.1f}mm, "
            f"置信度 {f.confidence_score:.2%}）"
        )

    output = DetectionOutput(
        tool_id=100,
        tool_name="肺结节检测 (MONAI RetinaNet)",
        image_path=image_path,
        mask_path=mask_path,
        findings=[f.model_dump() for f in findings],
        total_nodules=len(findings),
        summary="；".join(summary_parts),
    )

    print(f"[Tool 100] ✅ 检测完成，发现 {len(findings)} 枚结节")
    return output.model_dump()


def _prepare_image_for_monai(image_path: str) -> str:
    """将输入图像准备为 MONAI 可直接读取的格式。

    如果是 .zip (DICOM)，先转换为临时 NIfTI 文件。
    如果是 .nii.gz / .mhd，直接返回原路径。
    """
    lower = image_path.lower()

    if lower.endswith(".zip"):
        # 先用 SimpleITK 加载 DICOM 序列，再保存为 NIfTI
        import SimpleITK as sitk
        sitk_image = load_medical_image(image_path)
        tmp_nifti = os.path.join(
            tempfile.mkdtemp(prefix="medclaw_nifti_"), "ct_image.nii.gz"
        )
        sitk.WriteImage(sitk_image, tmp_nifti)
        print(f"[ImageLoader] DICOM → NIfTI 转换完成: {tmp_nifti}")
        return tmp_nifti

    return image_path


def _filter_by_lung_mask(
    results: list[dict],
    mask_path: str,
    image_path: str,
) -> list[dict]:
    """使用肺 mask 过滤掉不在肺实质内的假阳性检测结果。

    原理: 将每个结节的世界坐标转换回体素坐标，
    检查该体素在 mask 中是否标记为肺组织 (值 > 0)。
    """
    try:
        import SimpleITK as sitk

        mask_img = sitk.ReadImage(mask_path)
        mask_arr = sitk.GetArrayFromImage(mask_img)

        filtered = []
        for r in results:
            cx, cy, cz = r["center"]
            # 世界坐标 → 体素坐标 (使用 mask 图像的变换矩阵)
            # TransformPhysicalPointToIndex 实现 world → voxel 转换:
            #   voxel = Direction^(-1) × ((world - Origin) / Spacing)
            voxel_idx = mask_img.TransformPhysicalPointToIndex(
                [float(cx), float(cy), float(cz)]
            )

            # 检查是否在图像范围内
            size = mask_img.GetSize()  # (x, y, z)
            ix, iy, iz = voxel_idx
            if 0 <= ix < size[0] and 0 <= iy < size[1] and 0 <= iz < size[2]:
                # SimpleITK → numpy 时轴序反转: [z, y, x]
                if mask_arr[iz, iy, ix] > 0:
                    filtered.append(r)
                else:
                    print(
                        f"[Filter] 过滤掉肺外结节: "
                        f"({cx:.1f}, {cy:.1f}, {cz:.1f})"
                    )
            else:
                print(
                    f"[Filter] 过滤掉超范围结节: "
                    f"({cx:.1f}, {cy:.1f}, {cz:.1f})"
                )

        print(f"[Filter] 过滤结果: {len(results)} → {len(filtered)} 枚结节")
        return filtered

    except Exception as e:
        print(f"[Filter] ⚠️ Mask 过滤失败，保留所有结果: {e}")
        return results


# ============================================================
# Mock 兜底函数 — 当依赖库不可用时使用
# ============================================================

def _mock_detect_nodules(image_path: str, mask_path: Optional[str] = None) -> dict:
    """Mock 结节检测结果，当 MONAI 不可用时兜底。"""
    time.sleep(1)

    findings = [
        NoduleFinding(
            nodule_id=1,
            coordinates_mm=[-120.3, 45.7, -210.1],
            diameter_mm=12.5,
            confidence_score=0.92,
        ),
        NoduleFinding(
            nodule_id=2,
            coordinates_mm=[88.1, -32.4, -185.6],
            diameter_mm=4.2,
            confidence_score=0.78,
        ),
    ]

    output = DetectionOutput(
        tool_id=100,
        tool_name="肺结节检测（Mock）",
        image_path=image_path,
        mask_path=mask_path,
        findings=[f.model_dump() for f in findings],
        total_nodules=len(findings),
        summary=(
            "共检测到 2 枚肺结节（Mock 数据）。"
            "结节 #1（直径 12.5mm, 置信度 92%）；"
            "结节 #2（直径 4.2mm, 置信度 78%）。"
        ),
    )
    return output.model_dump()


# ============================================================
# 默认注册表 — 预加载真实工具
# ============================================================

def create_default_registry() -> ToolRegistry:
    """创建并返回预加载了工具的默认注册表。

    工具 ID 映射:
        100 → 肺结节检测 (detect_nodules)
        101 → 肺部分割 (segment_lungs)
    """
    registry = ToolRegistry()
    registry.register(100, detect_nodules)
    registry.register(101, segment_lungs)
    return registry
