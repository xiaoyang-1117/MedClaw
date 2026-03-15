"""
llm_reasoner.py — LLM 定性推理代理
====================================

负责调用 OpenAI 兼容的大模型 API，将量化分析结果转化为面向患者的
专业但易懂的医学总结报告。

API 配置完全从环境变量读取，支持 OpenAI / Qwen / Gemini 等任何
兼容 OpenAI Chat Completions 接口的模型。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI


# ============================================================
# System Prompt — 医学报告生成专家
# ============================================================

SYSTEM_PROMPT = """你是一位资深的医学影像 AI 助手，拥有放射科和临床医学双重专业背景。

你的职责是：
1. 接收来自自动化影像分析工具的 **量化结果**（JSON 格式）。
2. 根据任务描述和患者的原始提问，撰写一份 **结构化的医学分析报告**。

报告撰写规范：
- 语言风格：专业但对患者友好，避免过度使用专业术语；必要时附带通俗解释。
- 结构：使用 Markdown 格式，包含「影像发现」「临床意义」「建议」等段落。
- 如果量化结果中包含恶性概率或风险评级，需明确说明其含义和后续建议。
- **免责声明**：在报告末尾注明"本报告由 AI 辅助生成，仅供参考，不构成医疗诊断。请咨询专业医生获取最终诊断意见。"
- 使用中文撰写报告。
"""


class LLMReasoner:
    """封装 OpenAI 兼容 API 的定性推理引擎。

    从环境变量读取配置:
        - LLM_API_KEY   (必须)
        - LLM_BASE_URL  (可选，留空则使用 OpenAI 官方地址)
        - LLM_MODEL_NAME (必须)
    """

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.environ.get("LLM_API_KEY")
        self.base_url: Optional[str] = os.environ.get("LLM_BASE_URL") or None
        self.model_name: str = os.environ.get("LLM_MODEL_NAME", "qwen-plus")

        if not self.api_key:
            print("[LLMReasoner] ⚠️  未检测到 LLM_API_KEY，定性推理将不可用。")
            self.client = None
        else:
            # 如果 base_url 为空 / None，不传入该参数以兼容 OpenAI 默认地址
            init_kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                init_kwargs["base_url"] = self.base_url
            self.client = OpenAI(**init_kwargs)

    # --------------------------------------------------------
    # 核心推理方法
    # --------------------------------------------------------

    def qualitative_reasoning(
        self,
        action_desc: str,
        context_data: dict,
        user_query: str,
    ) -> str:
        """根据量化分析结果和用户提问，调用 LLM 生成定性医学报告。

        Args:
            action_desc:  Plan JSON 中的 action 字段描述 (如 "summarize nodule findings ...").
            context_data: 前序步骤的量化分析结果 (dict)，会被序列化为 JSON 字符串。
            user_query:   用户的原始自然语言提问。

        Returns:
            LLM 生成的医学报告文本 (Markdown 格式)。
            如果 API Key 未配置或调用失败，返回一段说明性文本。
        """
        if self.client is None:
            return self._fallback_report(action_desc, context_data, user_query)

        # 组装 User Prompt
        user_prompt = self._build_user_prompt(action_desc, context_data, user_query)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # 医疗场景倾向低温度以获得更稳定的输出
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            print(f"[LLMReasoner] ❌ API 调用失败: {e}")
            return self._fallback_report(action_desc, context_data, user_query)

    # --------------------------------------------------------
    # 内部辅助方法
    # --------------------------------------------------------

    @staticmethod
    def _build_user_prompt(
        action_desc: str,
        context_data: dict,
        user_query: str,
    ) -> str:
        """组装发送给 LLM 的 User Prompt。"""
        context_json = json.dumps(context_data, ensure_ascii=False, indent=2)

        return (
            f"## 任务描述\n{action_desc}\n\n"
            f"## 患者原始提问\n{user_query}\n\n"
            f"## 量化分析结果 (JSON)\n```json\n{context_json}\n```\n\n"
            "请根据以上信息，按照系统提示的规范撰写一份完整的医学分析报告。"
        )

    @staticmethod
    def _fallback_report(
        action_desc: str,
        context_data: dict,
        user_query: str,
    ) -> str:
        """当 LLM API 不可用时，生成一份基于模板的兜底报告。"""
        context_json = json.dumps(context_data, ensure_ascii=False, indent=2)

        return (
            "## 📋 医学分析报告（离线模式）\n\n"
            "> ⚠️ 当前未配置 LLM API Key 或 API 调用失败，以下为量化分析结果的原始展示。\n\n"
            f"### 任务描述\n{action_desc}\n\n"
            f"### 患者提问\n{user_query}\n\n"
            f"### 量化分析结果\n```json\n{context_json}\n```\n\n"
            "---\n"
            "*本报告由 AI 辅助生成，仅供参考，不构成医疗诊断。请咨询专业医生获取最终诊断意见。*"
        )
