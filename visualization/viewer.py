"""
viewer.py — Plotly CT 切片查看器 (Streamlit 集成)
====================================================

职责:
1. 使用 Plotly Heatmap 渲染单张 CT 切片 (肺窗)
2. 叠加结节 bounding box 标注
3. 提供 Streamlit 控件: 切片滚动条、置信度滑块、窗位/窗宽选择
4. 显示结节信息卡片

性能优化:
    - 仅渲染当前切片 (不加载整个 3D 体)
    - 使用 plotly.graph_objects 而非 plotly.express (更快)
    - 窗值映射在 numpy 层完成 (向量化)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .ct_loader import CTVolume, apply_window
from .nodule_overlay import (
    NoduleAnnotation,
    get_visible_nodules,
    make_plotly_box,
    prepare_nodule_annotations,
)


# 预定义窗位/窗宽组合
WINDOW_PRESETS: Dict[str, tuple] = {
    "🫁 肺窗": (-600, 1500),
    "🫀 纵隔窗": (40, 400),
    "🦴 骨窗": (400, 1800),
    "🧠 软组织窗": (50, 350),
}


def _go_to_slice(state_key: str, slice_idx: int) -> None:
    """Streamlit callback: 更新 session state 以跳转到指定切片。"""
    st.session_state[state_key] = slice_idx


def render_ct_viewer(
    ct_volume: CTVolume,
    findings: Optional[List[dict]] = None,
    key_prefix: str = "ct_viewer",
) -> None:
    """在 Streamlit 中渲染交互式 CT 切片查看器。

    功能:
        - 切片滚动 (slider)
        - 窗位/窗宽切换
        - 结节 bounding box 叠加
        - 置信度过滤
        - 结节信息卡片

    Args:
        ct_volume: 已加载的 CT 体数据
        findings:  结节检测结果列表 (可选)
        key_prefix: Streamlit widget key 前缀 (避免重复)
    """
    st.markdown("### 🖼️ CT 影像查看器")

    # --------------------------------------------------------
    # 控件区: 切片滑块 + 窗值选择 + 置信度过滤
    # --------------------------------------------------------
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 2, 2])

    with ctrl_col1:
        current_slice = st.slider(
            "切片位置 (Z轴)",
            min_value=0,
            max_value=ct_volume.num_slices - 1,
            value=ct_volume.num_slices // 2,
            key=f"{key_prefix}_slice",
            help=f"共 {ct_volume.num_slices} 张切片",
        )

    with ctrl_col2:
        window_name = st.selectbox(
            "窗位/窗宽",
            options=list(WINDOW_PRESETS.keys()),
            index=0,
            key=f"{key_prefix}_window",
        )
        wc, ww = WINDOW_PRESETS[window_name]

    with ctrl_col3:
        if findings:
            confidence_threshold = st.slider(
                "置信度过滤",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                key=f"{key_prefix}_conf",
            )
        else:
            confidence_threshold = 0.0

    # --------------------------------------------------------
    # 准备切片数据
    # --------------------------------------------------------
    slice_hu = ct_volume.array[current_slice]      # [Y, X] HU 值
    slice_display = apply_window(slice_hu, wc, ww)  # [Y, X] 0-255

    # --------------------------------------------------------
    # 准备结节标注
    # --------------------------------------------------------
    annotations: List[NoduleAnnotation] = []
    visible_nodules: List[NoduleAnnotation] = []

    if findings:
        annotations = prepare_nodule_annotations(
            findings, ct_volume, confidence_threshold
        )
        visible_nodules = get_visible_nodules(annotations, current_slice)

    # --------------------------------------------------------
    # Plotly 渲染
    # --------------------------------------------------------
    fig = go.Figure()

    # CT 切片: 使用 Heatmap (比 Image 更快，支持自定义 colorscale)
    fig.add_trace(go.Heatmap(
        z=slice_display,
        colorscale="gray",
        showscale=False,
        hovertemplate=(
            "X: %{x}<br>"
            "Y: %{y}<br>"
            "HU: %{customdata:.0f}<br>"
            "<extra></extra>"
        ),
        customdata=slice_hu,
    ))

    # 叠加结节 bounding box
    shapes = []
    plotly_annotations = []

    for nodule in visible_nodules:
        box = make_plotly_box(nodule, current_slice)
        shapes.append(box)

        # 标注文字: 结节编号 + 置信度
        vx = nodule.voxel_coord[0]
        vy = nodule.voxel_coord[1] - nodule.diameter_voxel[1] / 2.0 - 5
        plotly_annotations.append(dict(
            x=vx,
            y=max(vy, 0),
            text=f"#{nodule.nodule_id} ({nodule.confidence:.0%})",
            showarrow=False,
            font=dict(color="white", size=11),
            bgcolor="rgba(255, 50, 50, 0.7)",
            borderpad=2,
        ))

    # 布局
    fig.update_layout(
        shapes=shapes,
        annotations=plotly_annotations,
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=None,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            autorange="reversed",  # 医学图像惯例: Y 轴翻转
            title=None,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=560,
        plot_bgcolor="black",
        paper_bgcolor="black",
    )

    # 在 Streamlit 中渲染 Plotly 图表
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"{key_prefix}_plot",
        config={"scrollZoom": False},  # 禁用滚轮缩放，防止与页面滚动冲突导致抖动
    )

    # --------------------------------------------------------
    # 切片信息 + 结节卡片
    # --------------------------------------------------------
    info_col1, info_col2 = st.columns([1, 1])

    with info_col1:
        st.caption(
            f"📐 切片 {current_slice}/{ct_volume.num_slices - 1}　|　"
            f"尺寸 {ct_volume.width}×{ct_volume.height}　|　"
            f"间距 {ct_volume.spacing[0]:.3f}×{ct_volume.spacing[1]:.3f}×{ct_volume.spacing[2]:.3f} mm"
        )

    with info_col2:
        if visible_nodules:
            st.caption(
                f"🔴 当前切片附近可见 **{len(visible_nodules)}** 枚结节　|　"
                f"共 {len(annotations)} 枚 (≥{confidence_threshold:.0%})"
            )
        elif findings:
            st.caption(
                f"当前切片无可见结节　|　"
                f"共 {len(annotations)} 枚 (≥{confidence_threshold:.0%})"
            )

    # 结节信息卡片
    if visible_nodules:
        st.markdown("#### 📋 当前切片附近的结节")
        cols = st.columns(min(len(visible_nodules), 3))
        for idx, nodule in enumerate(visible_nodules[:6]):
            with cols[idx % 3]:
                _render_nodule_card(nodule)

    # --------------------------------------------------------
    # 结节快速导航列表
    # --------------------------------------------------------
    if annotations:
        with st.expander("🔍 结节检测列表", expanded=True):
            st.caption("👈 点击下方结节即可快速跳转至其中心切片位置：")
            nav_cols = st.columns(4)
            for idx, nodule in enumerate(annotations):
                with nav_cols[idx % 4]:
                    btn_label = f"#{nodule.nodule_id} - Z:{nodule.center_slice} - Conf:{nodule.confidence:.0%}"
                    st.button(
                        label=btn_label,
                        key=f"{key_prefix}_nav_{nodule.nodule_id}",
                        on_click=_go_to_slice,
                        args=(f"{key_prefix}_slice", nodule.center_slice),
                        use_container_width=True,
                    )


def _render_nodule_card(nodule: NoduleAnnotation) -> None:
    """渲染单个结节的信息卡片。"""
    # 置信度颜色
    if nodule.confidence >= 0.8:
        badge = "🔴"
        risk = "高"
    elif nodule.confidence >= 0.5:
        badge = "🟠"
        risk = "中"
    else:
        badge = "🟡"
        risk = "低"

    st.markdown(
        f"""
        <div style="
            background: #1e1e2e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
            font-size: 0.85rem;
            color: #e0e0e0;
        ">
            <div style="font-weight:700; margin-bottom:4px;">
                {badge} 结节 #{nodule.nodule_id}
            </div>
            <div>📏 直径: {nodule.diameter_mm:.1f} mm</div>
            <div>📍 位置: ({nodule.world_coord[0]:.1f}, {nodule.world_coord[1]:.1f}, {nodule.world_coord[2]:.1f})</div>
            <div>🎯 置信度: {nodule.confidence:.1%} ({risk})</div>
            <div>🔢 中心切片: Z={nodule.center_slice}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
