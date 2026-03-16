"""
app.py — MedClaw 前端交互界面 (Phase 2 · 重构版)
====================================================

基于 Streamlit 构建的类 ChatGPT 医学影像 AI 助手。

核心 UX 设计:
1. st.chat_input 固定在页面底部，历史消息瀑布流滚动
2. ➕ Popover 附件按钮 + 文件预览卡片 (带删除按钮)
3. 发送后严格清理: 递增 file_uploader key → 强制重置文件缓冲区
4. 友好的 Toast 提示替代红色警告边框

=== 状态清理机制详解 ===

Streamlit 的 st.file_uploader 在用户选择文件后会将文件对象缓存在
内部 WidgetState 中，绑定到 widget 的 key。清除它有且只有一种可靠方式:
**更换 key**。

流程:
  1. 用户选择文件 → 文件保存到临时目录 → pending_file 记录路径和元信息
  2. 用户按回车发送 → chat_input 返回文本
  3. 在同一次 rerun 中:
     a. 将 pending_file 中的路径拷贝到 submitted_file (一次性消费)
     b. 将 pending_file 设为 None
     c. 递增 uploader_key_counter → 下一次 rerun 时 file_uploader
        使用新 key → 老的文件缓冲区被 Streamlit 自动垃圾回收
  4. 后端处理使用 submitted_file 中的路径
  5. 处理完成后 submitted_file 也被清除

这保证:
  - 大文件不会驻留在 session_state 中
  - 下一轮纯文本追问绝不会重复发送上一轮的影像

启动命令:
    conda activate medai
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from executor import PlanExecutor, StepStatus
from llm_reasoner import LLMReasoner
from tools import create_default_registry
from visualization import load_ct_volume, render_ct_viewer


# ============================================================
# 页面配置 (必须在所有 st.xxx 调用之前)
# ============================================================

st.set_page_config(
    page_title="MedClaw — 医学影像 AI 助手",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 自定义 CSS — 聊天 UI 优化
# ============================================================

st.markdown("""
<style>
/* ---- 全局布局 ---- */
.main .block-container {
    max-width: 850px;
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}

/* ---- 标题区 ---- */
.app-header {
    text-align: center;
    padding: 1.2rem 0 0.6rem;
}
.app-header h1 { font-size: 1.5rem; font-weight: 700; margin: 0; }
.app-header p { color: #888; font-size: 0.88rem; margin-top: 0.2rem; }

/* ---- 执行步骤块 ---- */
.step-block {
    background: #f7f7f8;
    border-left: 4px solid #10a37f;
    padding: 0.5rem 0.9rem;
    margin: 0.3rem 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
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

/* ---- 文件预览卡片 ---- */
.file-preview-card {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #f0f4f8;
    border: 1px solid #d0d7de;
    border-radius: 10px;
    padding: 0.45rem 0.9rem;
    font-size: 0.82rem;
    color: #1f2937;
    margin: 0.3rem 0 0.5rem;
}
.file-preview-card .file-icon { font-size: 1.1rem; }
.file-preview-card .file-name { font-weight: 600; }
.file-preview-card .file-size { color: #6b7280; }

/* ---- 附件徽标 (聊天记录中) ---- */
.attachment-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #e8f5e9;
    border: 1px solid #a5d6a7;
    border-radius: 20px;
    padding: 0.25rem 0.7rem;
    font-size: 0.8rem;
    color: #2e7d32;
    margin-bottom: 0.4rem;
}

/* ---- 隐藏 file_uploader 的默认标签 ---- */
[data-testid="stFileUploader"] > label { display: none !important; }

/* ---- 移除 chat_input 红色/彩色边框 (目标 4: UI 细节优化) ---- */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] {
    border-color: #d0d5dd !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #86b7fe !important;
    box-shadow: 0 0 0 2px rgba(134, 183, 254, 0.25) !important;
}
/* 移除所有 Streamlit 默认的红色 invalid 边框 */
textarea:invalid, input:invalid {
    border-color: #d0d5dd !important;
    box-shadow: none !important;
}

/* ---- 底部输入区: 附件按钮与聊天框紧凑排列 ---- */
.bottom-input-area {
    margin-bottom: -0.5rem;
    padding-top: 0.3rem;
}
/* 让附件 popover 按钮在列内垂直居中 */
.bottom-input-area [data-testid="stPopover"] button {
    height: 2.6rem;
    width: 2.6rem;
    min-width: 2.6rem;
    padding: 0;
    font-size: 1.2rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #d0d5dd;
    background: #f9fafb;
    cursor: pointer;
    transition: background 0.15s;
}
.bottom-input-area [data-testid="stPopover"] button:hover {
    background: #e8f5e9;
    border-color: #a5d6a7;
}
</style>
""", unsafe_allow_html=True)


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
        "action": "summarize lung segmentation and nodule detection findings, "
                  "provide clinical interpretation and follow-up recommendations",
        "input_type": [],
        "output_type": "text",
        "depends_on": [1, 2],
    },
]


# ============================================================
# Session State 初始化
# ============================================================
#
# 关键状态变量:
#   messages           — 聊天历史 [{role, content}]
#   pending_file       — 当前待发送的文件 {name, size, path} 或 None
#   uploader_key       — file_uploader 的动态 key（递增以强制清除）
#   ct_volume          — 缓存的 CT 体数据（用于可视化持久渲染）
#   detection_findings — 缓存的结节检测结果

_DEFAULTS = {
    "messages": [],
    "pending_file": None,
    "uploader_key": 0,
    "tool_registry": None,
    "llm_reasoner": None,
    "ct_volume": None,
    "detection_findings": None,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# 延迟初始化重量级对象
if st.session_state.tool_registry is None:
    st.session_state.tool_registry = create_default_registry()
if st.session_state.llm_reasoner is None:
    st.session_state.llm_reasoner = LLMReasoner()


# ============================================================
# 辅助函数
# ============================================================

def _format_size(size_bytes: int) -> str:
    """将字节数格式化为可读的文件大小。"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _save_uploaded_file(uploaded_file) -> dict:
    """将 Streamlit UploadedFile 保存到临时目录，返回文件元信息。

    返回:
        {"name": str, "size": str, "path": str}
    """
    upload_dir = Path(tempfile.gettempdir()) / "medclaw_uploads"
    upload_dir.mkdir(exist_ok=True)
    save_path = upload_dir / uploaded_file.name
    save_path.write_bytes(uploaded_file.getvalue())
    return {
        "name": uploaded_file.name,
        "size": _format_size(uploaded_file.size),
        "path": str(save_path),
    }


def _clear_file_state():
    """清除文件相关状态 + 递增 uploader key 以强制重置组件。

    这是状态清理的核心函数。递增 uploader_key 后，
    下一次 rerun 中 st.file_uploader 会使用新 key 创建，
    旧 key 对应的文件缓冲区被 Streamlit 自动回收。
    """
    st.session_state.pending_file = None
    st.session_state.uploader_key += 1


# ============================================================
# 侧边栏 — LLM 配置 + Plan JSON
# ============================================================

with st.sidebar:
    st.markdown("### 🏥 MedClaw")
    st.caption("医学影像 AI 分析助手 · v0.3.0")
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
        st.caption("请在 `.env` 文件中设置")

    st.divider()
    st.markdown("#### 📋 执行计划")
    with st.expander("查看 Plan JSON"):
        st.json(PLAN_JSON)

    st.caption("v0.3.0 · Real Pipeline · lungmask + MONAI")


# ============================================================
# 标题
# ============================================================

st.markdown("""
<div class="app-header">
    <h1>🏥 MedClaw</h1>
    <p>上传胸部 CT 影像，输入您的问题，获取 AI 辅助分析报告</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 渲染历史消息 (瀑布流，最新在最下面)
# ============================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)


# ============================================================
# CT 影像查看器 — 持久渲染
# (放在聊天历史和输入区之间，不受 user_input rerun 影响)
# ============================================================

if st.session_state.ct_volume is not None:
    st.divider()
    try:
        render_ct_viewer(
            ct_volume=st.session_state.ct_volume,
            findings=st.session_state.detection_findings,
        )
    except Exception as e:
        st.warning(f"⚠️ CT 可视化渲染失败: {e}")


# ============================================================
# 底部输入区: 📎 附件按钮 + 文件预览卡片 + 聊天输入框
# ============================================================
# 设计: 使用 columns 将 📎 按钮放在左侧，文件预览卡片放在右侧，
# 紧贴 st.chat_input 上方，视觉上形成一体化底部输入区。
# st.chat_input 必须在顶层作用域调用 (Streamlit 硬约束)。

# --- 底部附件行: 📎 按钮 (左) + 文件预览 (右) ---
st.markdown('<div class="bottom-input-area">', unsafe_allow_html=True)
attach_col, preview_col = st.columns([1, 12])

with attach_col:
    # ▼ 紧凑型 📎 popover 按钮 ▼
    with st.popover("📎", use_container_width=True, help="附加医学影像"):
        st.markdown("**📎 附加医学影像**")
        st.caption("支持 .zip (DICOM序列), .nii.gz, .mhd+.raw (需同时选择)")

        # ▼ 关键: 使用动态 key，递增后组件重建，文件缓冲区自动清除 ▼
        uploader_key = f"file_uploader_{st.session_state.uploader_key}"

        uploaded_files = st.file_uploader(
            "选择影像文件",
            type=["zip", "gz", "nii", "mhd", "raw"],
            key=uploader_key,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )

        # 文件选择后立即保存到临时目录，记录到 pending_file
        if uploaded_files and st.session_state.pending_file is None:
            upload_dir = Path(tempfile.gettempdir()) / "medclaw_uploads"
            upload_dir.mkdir(exist_ok=True)

            saved_names = []
            primary_path = None
            total_size = 0

            for uf in uploaded_files:
                save_path = upload_dir / uf.name
                save_path.write_bytes(uf.getvalue())
                saved_names.append(uf.name)
                total_size += uf.size

            # 自动选择主文件: .mhd > .nii/.nii.gz > .zip
            for uf in uploaded_files:
                if uf.name.lower().endswith(".mhd"):
                    primary_path = str(upload_dir / uf.name)
                    break
            if primary_path is None:
                for uf in uploaded_files:
                    if uf.name.lower().endswith((".nii", ".nii.gz", ".gz")):
                        primary_path = str(upload_dir / uf.name)
                        break
            if primary_path is None:
                for uf in uploaded_files:
                    if uf.name.lower().endswith(".zip"):
                        primary_path = str(upload_dir / uf.name)
                        break
            if primary_path is None and saved_names:
                primary_path = str(upload_dir / saved_names[0])

            st.session_state.pending_file = {
                "name": ", ".join(saved_names),
                "size": _format_size(total_size),
                "path": primary_path,
            }
            st.rerun()  # 立即 rerun 以显示预览卡片

with preview_col:
    # --- 文件预览卡片 (已附加文件时显示) ---
    if st.session_state.pending_file is not None:
        pf = st.session_state.pending_file
        card_col, remove_col = st.columns([10, 1])
        with card_col:
            st.markdown(
                f'<div class="file-preview-card">'
                f'<span class="file-icon">📎</span>'
                f'<span class="file-name">{pf["name"]}</span>'
                f'<span class="file-size">({pf["size"]})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with remove_col:
            if st.button("✕", key="remove_file", help="移除附件"):
                _clear_file_state()
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# --- 聊天输入框 ---
# ▼ 关键: st.chat_input 在顶层作用域调用，Streamlit 自动固定在页面最底部 ▼
user_input = st.chat_input(
    "请输入您的问题，例如：帮我看看这个胸部CT有没有结节…"
)


# ============================================================
# 用户输入处理 — 发送消息 + 执行分析 + 状态清理
# ============================================================

if user_input:
    # ------------------------------------------------
    # 步骤 A: 捕获当前文件状态 (一次性消费)
    # ------------------------------------------------
    # 在清理前先保存路径，因为清理后 pending_file 就是 None 了
    submitted_file = st.session_state.pending_file
    submitted_path = submitted_file["path"] if submitted_file else None
    submitted_name = submitted_file["name"] if submitted_file else None

    # ------------------------------------------------
    # 步骤 B: 立即清理文件状态 (防止重复发送)
    # ------------------------------------------------
    # ▼ 关键: 在处理之前就清理，这样即使处理过程中 rerun，
    # 文件也不会被重复提交 ▼
    _clear_file_state()

    # ------------------------------------------------
    # 步骤 C: 添加用户消息到聊天历史
    # ------------------------------------------------
    # 如果有附件，先显示附件标签
    if submitted_name:
        attach_html = (
            f'<div class="attachment-badge">'
            f'📎 {submitted_name}'
            f'</div>'
        )
        st.session_state.messages.append(
            {"role": "user", "content": attach_html}
        )
        with st.chat_message("user"):
            st.markdown(attach_html, unsafe_allow_html=True)

    # 显示用户文本消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ------------------------------------------------
    # 步骤 D: AI 响应 — 执行分析管线
    # ------------------------------------------------
    with st.chat_message("assistant"):
        # D.1 校验是否有影像
        if not submitted_path:
            # 没有文件 → 友好提示 (Toast + 聊天消息，不使用红框)
            st.toast("💡 提示: 您可以点击左下角 ➕ 按钮附加 CT 影像文件", icon="📎")
            hint_msg = (
                "💡 未检测到附加的影像文件。如需影像分析，请点击输入框左侧的 "
                "**➕** 按钮上传胸部 CT（支持 .zip / .nii.gz / .mhd+.raw 格式），"
                "然后重新提问。\n\n"
                "如果您只是想进行文字咨询，请直接输入您的问题。"
            )
            st.info(hint_msg, icon="ℹ️")
            st.session_state.messages.append(
                {"role": "assistant", "content": hint_msg}
            )
        else:
            # D.2 创建执行器
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

            # D.3 流式展示执行步骤
            final_report = ""
            detection_findings = []

            st.markdown("**🔄 执行进度：**")

            for status in executor.execute(
                image_path=submitted_path,
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
                    if isinstance(status.result, dict):
                        with st.expander(
                            f"📊 步骤 {status.step_id} 量化结果",
                            expanded=False,
                        ):
                            st.json(status.result)
                        if "findings" in status.result:
                            detection_findings = status.result["findings"]
                            st.session_state.detection_findings = detection_findings
                    elif isinstance(status.result, str):
                        final_report = status.result
                elif status.status == StepStatus.FAILED:
                    st.markdown(
                        f'<div class="step-block failed">{status.message}</div>',
                        unsafe_allow_html=True,
                    )

            # D.4 缓存 CT 体数据 (用于可视化持久渲染)
            try:
                st.session_state.ct_volume = load_ct_volume(submitted_path)
            except Exception as e:
                st.caption(f"⚠️ CT 可视化加载失败: {e}")

            # D.5 展示最终报告
            if final_report:
                st.divider()
                st.markdown("**📝 最终分析报告：**")
                st.markdown(final_report)

            # D.6 组装完整消息存入历史
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

            # ▼ 关键修复: 强制 rerun 以立即渲染 CT 查看器 ▼
            # 影像处理完成后 ct_volume 已写入 session_state，
            # 但当前 run 的渲染流程已过（viewer 在消息历史后面渲染），
            # 必须 rerun 才能在下一轮 top-to-bottom 执行中渲染 viewer
            if st.session_state.ct_volume is not None:
                st.rerun()
