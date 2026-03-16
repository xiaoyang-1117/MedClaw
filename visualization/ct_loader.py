"""
ct_loader.py — CT 体数据加载与缓存
====================================

职责:
1. 使用 SimpleITK 加载各格式 CT 文件 (.mhd/.nii.gz/.zip)
2. 将 3D 体数据转换为 numpy 数组 (HU 窗值映射)
3. 提取空间元数据 (Spacing, Origin, Direction)
4. 通过 Streamlit @st.cache_data 缓存，避免重复加载

性能要求: 支持 (300, 512, 512) 体数据
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CTVolume:
    """CT 体数据容器。

    Attributes:
        array:     numpy 数组 [Z, Y, X]，HU 值或归一化后的像素值
        spacing:   体素间距 (sx, sy, sz) mm
        origin:    图像原点 (ox, oy, oz) mm
        direction: 方向余弦矩阵 (9 个元素)
        shape:     (num_slices, height, width)
    """
    array: np.ndarray       # [Z, Y, X]
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: Tuple[float, ...]
    shape: Tuple[int, int, int]

    @property
    def num_slices(self) -> int:
        return self.shape[0]

    @property
    def height(self) -> int:
        return self.shape[1]

    @property
    def width(self) -> int:
        return self.shape[2]


def load_ct_volume(file_path: str) -> CTVolume:
    """加载 CT 体数据并返回 CTVolume 对象。

    加载流程:
        1. 使用 tools.load_medical_image 加载 SimpleITK Image
        2. 提取 numpy 数组 (轴序: Z, Y, X)
        3. 提取空间元数据

    Args:
        file_path: CT 文件路径 (.mhd, .nii.gz, .zip 等)

    Returns:
        CTVolume 对象
    """
    import SimpleITK as sitk
    import sys
    import os

    # 添加项目根目录到 path 以导入 tools
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from tools import load_medical_image

    sitk_image = load_medical_image(file_path)

    # SimpleITK → numpy: 轴序自动从 (X,Y,Z) 变为 (Z,Y,X)
    array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)

    return CTVolume(
        array=array,
        spacing=sitk_image.GetSpacing(),     # (sx, sy, sz) in mm
        origin=sitk_image.GetOrigin(),       # (ox, oy, oz) in mm
        direction=sitk_image.GetDirection(),  # 9-element direction cosines
        shape=array.shape,                    # (Z, Y, X)
    )


def apply_window(
    slice_array: np.ndarray,
    window_center: float = -600.0,
    window_width: float = 1500.0,
) -> np.ndarray:
    """应用 CT 窗值/窗位映射。

    将 HU 值映射到 [0, 255] 显示范围。
    - 肺窗: center=-600, width=1500 (显示肺实质 + 结节)
    - 纵隔窗: center=40, width=400

    Args:
        slice_array: 2D numpy 数组 (HU 值)
        window_center: 窗位 (HU)
        window_width:  窗宽 (HU)

    Returns:
        [0, 255] 范围的 uint8 数组
    """
    lower = window_center - window_width / 2.0
    upper = window_center + window_width / 2.0

    # 裁剪到窗口范围，然后线性映射到 [0, 255]
    clipped = np.clip(slice_array, lower, upper)
    normalized = (clipped - lower) / (upper - lower)
    return (normalized * 255).astype(np.uint8)
