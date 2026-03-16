"""
nodule_overlay.py — 结节标注叠加层
====================================

职责:
1. 将结节世界坐标 (mm) 转换为体素坐标 (voxel index)
2. 计算每个结节在指定 Z 切片上的可见性
3. 生成 Plotly 兼容的矩形标注形状 (bounding box)
4. 支持按置信度过滤

坐标转换说明:
    世界坐标 → 体素坐标:
        voxel = (world - origin) / spacing
    注意: SimpleITK 的 Origin/Spacing 是 (X, Y, Z) 顺序。
    numpy 数组是 (Z, Y, X) 顺序。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .ct_loader import CTVolume


@dataclass
class NoduleAnnotation:
    """单个结节的标注信息 (体素坐标系)。"""
    nodule_id: int
    world_coord: Tuple[float, float, float]    # (x, y, z) mm
    voxel_coord: Tuple[float, float, float]    # (vx, vy, vz) voxel index
    diameter_mm: float
    diameter_voxel: Tuple[float, float, float] # 各轴上的直径 (voxel)
    confidence: float
    center_slice: int  # 结节中心所在的 Z 切片索引


def world_to_voxel(
    world_xyz: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """世界坐标 (mm) → 体素坐标 (voxel index)。

    简化公式 (忽略方向余弦矩阵的非对角元素):
        voxel_i = (world_i - origin_i) / spacing_i

    Args:
        world_xyz: (x, y, z) 世界坐标 mm
        origin:    (ox, oy, oz) 图像原点 mm
        spacing:   (sx, sy, sz) 体素间距 mm

    Returns:
        (vx, vy, vz) 体素坐标 (浮点数)
    """
    vx = (world_xyz[0] - origin[0]) / spacing[0]
    vy = (world_xyz[1] - origin[1]) / spacing[1]
    vz = (world_xyz[2] - origin[2]) / spacing[2]
    return (vx, vy, vz)


def prepare_nodule_annotations(
    findings: List[dict],
    ct_volume: CTVolume,
    confidence_threshold: float = 0.0,
) -> List[NoduleAnnotation]:
    """将检测结果转换为体素坐标系的标注列表。

    Args:
        findings:     结节检测结果列表 (含 coordinates_mm, diameter_mm, confidence_score)
        ct_volume:    CT 体数据
        confidence_threshold: 置信度过滤阈值

    Returns:
        NoduleAnnotation 列表 (已按置信度降序排列)
    """
    annotations = []
    spacing = ct_volume.spacing  # (sx, sy, sz)
    origin = ct_volume.origin    # (ox, oy, oz)

    for finding in findings:
        confidence = finding.get("confidence_score", 0.0)
        if confidence < confidence_threshold:
            continue

        world_coord = finding.get("coordinates_mm", [0, 0, 0])
        diameter_mm = finding.get("diameter_mm", 5.0)

        # 世界坐标 → 体素坐标
        # SimpleITK: origin/spacing 是 (X, Y, Z) 顺序
        vx, vy, vz = world_to_voxel(
            (world_coord[0], world_coord[1], world_coord[2]),
            origin,
            spacing,
        )

        # 直径从 mm 转换为各轴的 voxel 数
        dx_voxel = diameter_mm / spacing[0]
        dy_voxel = diameter_mm / spacing[1]
        dz_voxel = diameter_mm / spacing[2]

        # numpy 数组轴序: [Z, Y, X]
        # 结节中心的 Z 切片索引
        center_slice = int(round(vz))

        # 限制在有效范围内
        center_slice = max(0, min(center_slice, ct_volume.num_slices - 1))

        annotations.append(NoduleAnnotation(
            nodule_id=finding.get("nodule_id", len(annotations) + 1),
            world_coord=(world_coord[0], world_coord[1], world_coord[2]),
            voxel_coord=(vx, vy, vz),
            diameter_mm=diameter_mm,
            diameter_voxel=(dx_voxel, dy_voxel, dz_voxel),
            confidence=confidence,
            center_slice=center_slice,
        ))

    # 按置信度降序排列
    annotations.sort(key=lambda a: a.confidence, reverse=True)
    return annotations


def get_visible_nodules(
    annotations: List[NoduleAnnotation],
    current_slice: int,
    slice_tolerance: float = 3.0,
) -> List[NoduleAnnotation]:
    """筛选在当前切片附近可见的结节。

    一个结节在 Z 轴上跨越 [center_z - radius_z, center_z + radius_z] 范围。
    如果当前切片落在此范围 ± tolerance 内，则认为该结节可见。

    Args:
        annotations:     所有结节标注
        current_slice:   当前显示的 Z 切片索引
        slice_tolerance: 额外容差 (voxel 数)

    Returns:
        当前切片附近可见的结节列表
    """
    visible = []
    for ann in annotations:
        # 结节在 Z 轴上的半径 (voxel)
        radius_z = ann.diameter_voxel[2] / 2.0
        z_min = ann.voxel_coord[2] - radius_z - slice_tolerance
        z_max = ann.voxel_coord[2] + radius_z + slice_tolerance

        if z_min <= current_slice <= z_max:
            visible.append(ann)

    return visible


def make_plotly_box(
    annotation: NoduleAnnotation,
    current_slice: int,
) -> dict:
    """生成 Plotly layout.shapes 兼容的矩形标注。

    矩形框在 XY 平面上围绕结节中心画出。
    框的透明度根据当前切片与结节中心的距离动态调整:
    越靠近中心越不透明。

    注意:
        numpy 数组轴序是 [Z, Y, X]。
        Plotly heatmap 中，x 轴对应第2维(X)，y 轴对应第1维(Y)。
        所以矩形框用 vx (X方向) 和 vy (Y方向)。

    Returns:
        Plotly shape dict
    """
    vx, vy, vz = annotation.voxel_coord
    rx = annotation.diameter_voxel[0] / 2.0  # X 方向半径 (voxel)
    ry = annotation.diameter_voxel[1] / 2.0  # Y 方向半径 (voxel)

    # 动态透明度: 距离中心越远越透明
    dist = abs(current_slice - vz)
    max_dist = annotation.diameter_voxel[2] / 2.0 + 3.0
    alpha = max(0.2, 1.0 - dist / max_dist)

    # 根据置信度调整颜色强度
    if annotation.confidence >= 0.8:
        color = f"rgba(255, 50, 50, {alpha:.2f})"   # 高置信度: 红色
    elif annotation.confidence >= 0.5:
        color = f"rgba(255, 165, 0, {alpha:.2f})"   # 中置信度: 橙色
    else:
        color = f"rgba(255, 255, 0, {alpha:.2f})"   # 低置信度: 黄色

    return dict(
        type="rect",
        x0=vx - rx,
        y0=vy - ry,
        x1=vx + rx,
        y1=vy + ry,
        line=dict(color=color, width=2),
        fillcolor=color.replace(f"{alpha:.2f}", f"{alpha * 0.15:.2f}"),
        name=f"结节 #{annotation.nodule_id}",
    )
