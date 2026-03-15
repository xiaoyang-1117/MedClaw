"""
app.py — MedClaw 前端交互界面 (Phase 2)
==========================================

基于 Streamlit 构建的类 ChatGPT 医学影像 AI 助手。
功能:
1. 底部固定输入区 + ➕ 附件按钮 (popover)
2. 支持上传 .zip / .nii.gz / .mhd / .raw 格式
3. 三步 DAG 工作流: 肺分割 → 结节检测 → LLM 总结
4. 流式展示中间步骤和最终报告

启动命令:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from executor import PlanExecutor, StepStatus
from llm_reasoner import LLMReasoner
from tools import create_default_registry


# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="MedClaw — 医学影像 AI 助手",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 自定义样式 — 致敬 ChatGPT 风格
# ============================================================

st.markdown(
    """
    <style>
    /* ---- 全局 ---- */
    .main .block-container {
        max-width: 820px;
        padding-top: 1.5rem;
        padding-bottom: 5rem;
    }

    /* ---- 顶部标题 ---- */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    .app-header p {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    /* ---- 执行步骤块 ---- */
    .step-block {
        background: #f7f7f8;
        border-left: 4px solid #10a37f;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #333;
    }
    .step-block.failed {
        border-left-color: #ef4444;
        background: #fef2f2;
    }
    .step-block.running {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }

    /* ---- 附件提示 ---- */
    .attachment-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #e8f5e9;
        border: 1px solid #a5d6a7;
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.82rem;
        color: #2e7d32;
        margin-bottom: 0.5rem;
    }

    /* ---- 聊天气泡微调 ---- */
    [data-testid="stChatMessage"] {
        padding: 0.8rem 0;
    }

    /* ---- Popover 宽度 ---- */
    [data-testid="stPopover"] {
        min-width: 320px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 三步 Plan JSON — 肺分割 → 结节检测 → LLM 总结
# ============================================================

PLAN_JSON: list[dict] = [
    {
        "id": 1,
        "tool": [101],
        "action_type": "quantitative",
        "action": "segment lung parenchyma from chest CT",
        "input_type": [0],
        "output_type": "intermediate result",
        "output_path": "lung_mask.nii.gz",
    },
    {
        "id": 2,
        "tool": [100],
        "action_type": "quantitative",
        "action": "detect pulmonary nodules on chest CT using lung mask",
        "input_type": [0],
        "output_type": "intermediate result",
        "output_path": "nodules.json",
        "depends_on": [1],
    },
    {
        "id": 3,
        "tool": [],
        "action_type": "qualitative",
        "action": "summarize lung segmentation and nodule detection findings, provide clinical interpretation and follow-up recommendations",
        "input_type": [],
        "output_type": "text",
        "depends_on": [1, 2],
    },
]


# ============================================================
# 初始化 Session State
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "tool_registry" not in st.session_state:
    st.session_state.tool_registry = create_default_registry()

if "llm_reasoner" not in st.session_state:
    st.session_state.llm_reasoner = LLMReasoner()

if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None


# ============================================================
# 侧边栏 — LLM 配置状态
# ============================================================

with st.sidebar:
    st.markdown("### 🏥 MedClaw")
    st.caption("医学影像 AI 分析助手 · v0.2.0")
    st.divider()

    st.markdown("#### ⚙️ LLM 配置")
    api_key = os.environ.get("LLM_API_KEY", "")
    model_name = os.environ.get("LLM_MODEL_NAME", "未设置")
    base_url = os.environ.get("LLM_BASE_URL", "默认 (OpenAI)")

    if api_key and api_key != "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
        st.markdown(f"- 模型: `{model_name}`")
        st.markdown(f"- 地址: `{base_url}`")
        st.markdown("- 状态: 🟢 已配置")
    else:
        st.markdown("- 状态: 🔴 未配置")
        st.caption("请在 `.env` 文件中设置 API Key")

    st.divider()

    st.markdown("#### 📋 执行计划")
    with st.expander("查看 Plan JSON"):
        st.json(PLAN_JSON)

    st.divider()

    if st.session_state.uploaded_path:
        st.markdown("#### 📎 当前附件")
        st.caption(st.session_state.uploaded_name)

    st.caption("v0.2.0 · Real Pipeline · lungmask + MONAI")


# ============================================================
# 标题
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <h1>🏥 MedClaw</h1>
        <p>上传胸部 CT 影像，输入您的问题，获取 AI 辅助分析报告</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 渲染历史消息
# ============================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)


# ============================================================
# 底部输入区: ➕ 附件按钮 + 聊天输入框
# ============================================================

# 使用 columns 将 popover 和 chat_input 并排放置
col_attach, col_input = st.columns([1, 12])

with col_attach:
    with st.popover("➕", use_container_width=True):
        st.markdown("**📎 附加医学影像**")
        st.caption("支持 .zip (DICOM序列), .nii.gz, .mhd, .raw")
        uploaded_file = st.file_uploader(
            "选择影像文件",
            type=["zip", "gz", "nii", "mhd", "raw"],
            key="file_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            # 保存到临时目录
            upload_dir = Path(tempfile.gettempdir()) / "medclaw_uploads"
            upload_dir.mkdir(exist_ok=True)
            save_path = upload_dir / uploaded_file.name
            save_path.write_bytes(uploaded_file.getvalue())

            st.session_state.uploaded_path = str(save_path)
            st.session_state.uploaded_name = uploaded_file.name

            st.success(f"✅ {uploaded_file.name}")

with col_input:
    user_input = st.chat_input(
        "请输入您的问题，例如：帮我看看这个胸部CT有没有结节…"
    )


# ============================================================
# 用户输入处理
# ============================================================

if user_input:
    uploaded_path = st.session_state.uploaded_path
    uploaded_name = st.session_state.uploaded_name

    # 1. 如果有附件，先显示附件系统消息
    if uploaded_name:
        attach_msg = f'<div class="attachment-badge">📎 已附加影像文件：{uploaded_name}</div>'
        st.session_state.messages.append({"role": "user", "content": attach_msg})
        with st.chat_message("user"):
            st.markdown(attach_msg, unsafe_allow_html=True)

    # 2. 展示用户文本消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 3. AI 响应
    with st.chat_message("assistant"):
        # 3.1 校验是否有影像
        if not uploaded_path:
            no_file_msg = (
                "⚠️ 未检测到附加的影像文件。请点击输入框左侧的 **➕** 按钮上传"
                "胸部 CT 影像（支持 .zip / .nii.gz / .mhd 格式），然后重新提问。"
            )
            st.warning(no_file_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": no_file_msg}
            )
        else:
            # 3.2 创建执行器并运行
            try:
                executor = PlanExecutor(
                    plan_json=PLAN_JSON,
                    tool_registry=st.session_state.tool_registry,
                    llm_reasoner=st.session_state.llm_reasoner,
                )
            except Exception as e:
                st.error(f"❌ Plan JSON 校验失败: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"❌ Plan JSON 校验失败: {e}"}
                )
                st.stop()

            # 3.3 流式展示执行步骤
            final_report = ""

            st.markdown("**🔄 执行进度：**")

            for status in executor.execute(
                image_path=uploaded_path,
                user_query=user_input,
            ):
                if status.status == StepStatus.STARTED:
                    st.markdown(
                        f'<div class="step-block running">{status.message}</div>',
                        unsafe_allow_html=True,
                    )

                elif status.status == StepStatus.COMPLETED:
                    st.markdown(
                        f'<div class="step-block">{status.message}</div>',
                        unsafe_allow_html=True,
                    )
                    # 显示量化结果
                    if isinstance(status.result, dict):
                        with st.expander(
                            f"📊 步骤 {status.step_id} 量化结果", expanded=False
                        ):
                            st.json(status.result)
                    # 记录最终文本结果（定性报告）
                    elif isinstance(status.result, str):
                        final_report = status.result

                elif status.status == StepStatus.FAILED:
                    st.markdown(
                        f'<div class="step-block failed">{status.message}</div>',
                        unsafe_allow_html=True,
                    )

            # 3.4 展示最终报告
            if final_report:
                st.divider()
                st.markdown("**📝 最终分析报告：**")
                st.markdown(final_report)

            # 3.5 组装完整消息存入历史
            full_response = ""
            if final_report:
                full_response = final_report
            else:
                last_result = executor.get_final_result()
                if isinstance(last_result, dict):
                    full_response = (
                        "**量化分析结果：**\n```json\n"
                        + json.dumps(last_result, ensure_ascii=False, indent=2)
                        + "\n```"
                    )
                else:
                    full_response = str(last_result) if last_result else "分析完成。"

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
