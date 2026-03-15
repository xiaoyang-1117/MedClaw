"""
executor.py — DAG 任务调度引擎
================================

核心职责:
1. 使用 Pydantic Plan 模型校验 Plan JSON。
2. 基于 depends_on 字段进行拓扑排序 (Kahn's Algorithm)。
3. 按顺序执行每个任务节点，路由到 ToolRegistry 或 LLMReasoner。
4. 以生成器形式 yield 执行进度，方便前端流式展示。
5. 在多步骤工作流中自动传递中间结果 (如 mask_path)。
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Dict, Generator, List, Optional

from schemas import Plan, PlanNode
from llm_reasoner import LLMReasoner
from tools import ToolRegistry


# ============================================================
# 执行状态消息类型
# ============================================================

class StepStatus:
    """执行步骤的状态更新，用于 UI 层展示。"""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"

    def __init__(
        self,
        step_id: int,
        status: str,
        message: str,
        result: Any = None,
    ) -> None:
        self.step_id = step_id
        self.status = status
        self.message = message
        self.result = result


# ============================================================
# DAG 调度引擎
# ============================================================

class PlanExecutor:
    """基于拓扑排序的 DAG 任务调度器。

    接收 Plan JSON，使用 Pydantic 校验结构，解析依赖关系，
    按顺序执行每个步骤，将量化任务路由到 ToolRegistry，
    定性任务路由到 LLMReasoner。

    使用方式:
        executor = PlanExecutor(plan_json, registry, reasoner)
        for status in executor.execute(image_path, user_query):
            print(status.message)
    """

    def __init__(
        self,
        plan_json: list[dict],
        tool_registry: ToolRegistry,
        llm_reasoner: LLMReasoner,
    ) -> None:
        # 使用 Pydantic 模型强制校验 Plan JSON 结构
        self.plan = Plan(nodes=[PlanNode(**item) for item in plan_json])
        self.node_map: Dict[int, PlanNode] = {n.id: n for n in self.plan.nodes}
        self.tool_registry = tool_registry
        self.llm_reasoner = llm_reasoner
        # 执行上下文：存储每个步骤的输出
        self.context: Dict[int, Any] = {}

    # --------------------------------------------------------
    # 拓扑排序 — Kahn's Algorithm (BFS)
    # --------------------------------------------------------

    def _topological_sort(self) -> List[int]:
        """对任务节点进行拓扑排序。

        Returns:
            按依赖关系排列的 task id 列表。

        Raises:
            ValueError: 如果存在循环依赖。
        """
        nodes = self.plan.nodes

        # 构建入度表和邻接表
        in_degree: Dict[int, int] = {n.id: 0 for n in nodes}
        adjacency: Dict[int, List[int]] = {n.id: [] for n in nodes}

        for node in nodes:
            for dep_id in node.depends_on:
                if dep_id in adjacency:
                    adjacency[dep_id].append(node.id)
                    in_degree[node.id] += 1

        # BFS: 从入度为 0 的节点开始
        queue: deque[int] = deque()
        for node_id, degree in in_degree.items():
            if degree == 0:
                queue.append(node_id)

        sorted_ids: List[int] = []
        while queue:
            current = queue.popleft()
            sorted_ids.append(current)
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_ids) != len(nodes):
            raise ValueError(
                f"DAG 中存在循环依赖！已排序 {len(sorted_ids)} 个节点，"
                f"但共有 {len(nodes)} 个节点。"
            )

        return sorted_ids

    # --------------------------------------------------------
    # 执行引擎（生成器）
    # --------------------------------------------------------

    def execute(
        self,
        image_path: Optional[str],
        user_query: str,
    ) -> Generator[StepStatus, None, None]:
        """执行整个 Plan，按拓扑顺序逐步执行并 yield 进度。

        Args:
            image_path: 用户上传的医学影像路径（可选）。
            user_query: 用户的自然语言提问。

        Yields:
            StepStatus 对象，包含步骤 ID、状态、消息和结果。
        """
        # 1. 拓扑排序
        try:
            execution_order = self._topological_sort()
        except ValueError as e:
            yield StepStatus(
                step_id=-1,
                status=StepStatus.FAILED,
                message=f"❌ 调度失败: {e}",
            )
            return

        # 2. 按顺序执行
        for task_id in execution_order:
            node = self.node_map[task_id]

            yield StepStatus(
                step_id=task_id,
                status=StepStatus.STARTED,
                message=f"⚙️ 正在执行步骤 {task_id}: {node.action}",
            )

            try:
                if node.action_type.value == "quantitative":
                    result = self._execute_quantitative(node, image_path)
                elif node.action_type.value == "qualitative":
                    result = self._execute_qualitative(node, user_query)
                else:
                    result = {"warning": f"未知的 action_type: {node.action_type}"}

                self.context[task_id] = result

                yield StepStatus(
                    step_id=task_id,
                    status=StepStatus.COMPLETED,
                    message=f"✅ 步骤 {task_id} 完成: {node.action}",
                    result=result,
                )

            except Exception as e:
                error_msg = f"❌ 步骤 {task_id} 执行失败: {e}"
                self.context[task_id] = {"error": str(e)}

                yield StepStatus(
                    step_id=task_id,
                    status=StepStatus.FAILED,
                    message=error_msg,
                    result={"error": str(e)},
                )

    # --------------------------------------------------------
    # 量化任务执行
    # --------------------------------------------------------

    def _execute_quantitative(
        self,
        node: PlanNode,
        image_path: Optional[str],
    ) -> Any:
        """执行量化分析任务（调用 ToolRegistry）。

        自动从依赖步骤的输出中提取 mask_path 等上下文参数，
        实现多步骤工作流中的数据传递。
        """
        if not node.tool:
            return {"warning": "该量化任务未指定 tool ID"}

        # 从依赖步骤中收集额外参数 (如 mask_path)
        extra_kwargs: Dict[str, Any] = {}
        for dep_id in node.depends_on:
            dep_result = self.context.get(dep_id, {})
            if isinstance(dep_result, dict):
                # 如果前序步骤产生了 mask_path，自动传递给当前步骤
                if "mask_path" in dep_result:
                    extra_kwargs["mask_path"] = dep_result["mask_path"]

        # 对每个 tool ID 依次调用，合并结果
        combined_results: Dict[str, Any] = {}
        for tool_id in node.tool:
            result = self.tool_registry.call(
                tool_id,
                image_path=image_path or "no_image_provided",
                **extra_kwargs,
            )
            combined_results[f"tool_{tool_id}"] = result

        # 如果只有一个 tool，直接返回其结果
        if len(node.tool) == 1:
            return combined_results[f"tool_{node.tool[0]}"]

        return combined_results

    # --------------------------------------------------------
    # 定性任务执行
    # --------------------------------------------------------

    def _execute_qualitative(
        self,
        node: PlanNode,
        user_query: str,
    ) -> str:
        """执行定性分析任务（调用 LLMReasoner）。"""
        # 收集依赖步骤的输出作为上下文
        dep_context: Dict[str, Any] = {}
        for dep_id in node.depends_on:
            if dep_id in self.context:
                dep_context[f"step_{dep_id}_result"] = self.context[dep_id]

        return self.llm_reasoner.qualitative_reasoning(
            action_desc=node.action,
            context_data=dep_context,
            user_query=user_query,
        )

    # --------------------------------------------------------
    # 获取最终结果
    # --------------------------------------------------------

    def get_final_result(self) -> Any:
        """返回 DAG 中最后一个执行节点的结果。"""
        if not self.context:
            return None
        last_id = max(self.context.keys())
        return self.context[last_id]
