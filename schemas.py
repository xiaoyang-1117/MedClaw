"""
schemas.py — Pydantic 数据模型定义
====================================

定义所有用于 JSON 校验的 Pydantic 模型:
1. PlanNode / Plan     — 校验上游 Plan JSON 结构
2. NoduleFinding       — 结节检测的量化输出
3. SegmentationOutput  — 肺分割的量化输出
4. ToolOutput          — 统一的工具返回格式
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator


# ============================================================
# Plan JSON 数据模型
# ============================================================

class ActionType(str, Enum):
    """任务动作类型枚举。"""
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"


class PlanNode(BaseModel):
    """Plan JSON 中单个任务节点。

    字段说明:
        id:          任务唯一标识符
        tool:        量化任务对应的 Tool ID 列表
        action_type: "quantitative" 或 "qualitative"
        action:      自然语言描述的动作
        input_type:  输入类型 ID 列表
        output_type: 输出类型描述
        output_path: 可选的输出文件路径
        depends_on:  依赖的前序任务 ID 列表
    """
    id: int
    tool: List[int] = Field(default_factory=list)
    action_type: ActionType
    action: str
    input_type: List[int] = Field(default_factory=list)
    output_type: str = "text"
    output_path: Optional[str] = None
    depends_on: List[int] = Field(default_factory=list)


class Plan(BaseModel):
    """Plan JSON 的顶层包装器，支持结构校验。"""
    nodes: List[PlanNode]

    @model_validator(mode="after")
    def validate_dag_integrity(self) -> "Plan":
        """校验 DAG 完整性：所有 depends_on 引用的 ID 必须存在。"""
        node_ids = {n.id for n in self.nodes}
        for node in self.nodes:
            for dep_id in node.depends_on:
                if dep_id not in node_ids:
                    raise ValueError(
                        f"任务 {node.id} 依赖的任务 ID {dep_id} 不存在于计划中。"
                        f"已有的任务 ID: {sorted(node_ids)}"
                    )
        return self


# ============================================================
# 工具输出数据模型
# ============================================================

class NoduleFinding(BaseModel):
    """单个结节的检测结果。

    注意: 仅包含检测置信度 (confidence_score)，
    不包含恶性概率字段 — 恶性分类由独立模型完成。
    """
    nodule_id: int = Field(..., description="结节编号")
    coordinates_mm: List[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="结节中心点的世界坐标 [x, y, z]，单位 mm",
    )
    diameter_mm: float = Field(..., ge=0, description="结节估算直径，单位 mm")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="检测置信度 (0~1)"
    )


class SegmentationOutput(BaseModel):
    """肺分割工具的输出结果。"""
    tool_id: int
    tool_name: str = "肺部分割"
    image_path: str
    mask_path: str = Field(..., description="生成的肺 mask 文件路径")
    left_lung_volume_ml: Optional[float] = Field(None, description="左肺体积 (mL)")
    right_lung_volume_ml: Optional[float] = Field(None, description="右肺体积 (mL)")
    status: str = "success"


class DetectionOutput(BaseModel):
    """结节检测工具的输出结果。"""
    tool_id: int
    tool_name: str = "肺结节检测"
    image_path: str
    mask_path: Optional[str] = Field(None, description="使用的肺 mask 路径")
    findings: List[NoduleFinding] = Field(default_factory=list)
    total_nodules: int = Field(0, description="检测到的结节总数")
    summary: str = ""


class ToolOutput(BaseModel):
    """统一的工具输出包装器，兼容不同工具类型。"""
    tool_id: int
    tool_name: str
    status: str = "success"
    data: Any = Field(None, description="工具特定的输出数据")
    error: Optional[str] = None
