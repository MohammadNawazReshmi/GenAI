import os
import re
import sys
import uuid
import subprocess
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import agent tools from min.py
from min import calculator, say_hello, get_system_time

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def clean_response_text(text: str) -> str:
    """Sanitizes model response text by stripping internal reasoning tags (<think>...</think>)
    and whitespace formatting.
    """
    if not text:
        return ""
    # Remove completed <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove unclosed <think> blocks during streaming
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Grok AI Chatbot",
    page_icon="𝕏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom Grok CSS Theme Styling
# ---------------------------------------------------------------------------
GROK_CSS = """
<style>
/* Reset and dark theme base */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #09090b !important;
    color: #e4e4e7 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stHeader"] {
    background-color: rgba(9, 9, 11, 0.85) !important;
    backdrop-filter: blur(12px) !important;
}

[data-testid="stSidebar"] {
    background-color: #0d0d10 !important;
    border-right: 1px solid #1f1f23 !important;
}

/* Grok Navbar Header */
.grok-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background: linear-gradient(180deg, rgba(20, 20, 25, 0.9) 0%, rgba(13, 13, 16, 0.6) 100%);
    border: 1px solid #27272a;
    border-radius: 14px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.grok-logo {
    display: flex;
    align-items: center;
    gap: 12px;
}

.grok-logo-icon {
    font-size: 26px;
    font-weight: 800;
    color: #ffffff;
    background: linear-gradient(135deg, #ffffff 0%, #a1a1aa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}

.grok-badge {
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(96, 165, 250, 0.3);
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.grok-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: #a1a1aa;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 8px #22c55e;
}

/* Grok Chat Input Container */
[data-testid="stChatInput"] {
    background-color: #141417 !important;
    border: 1px solid #27272a !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
    color: #f4f4f5 !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #3f3f46 !important;
    box-shadow: 0 0 0 2px rgba(161, 161, 170, 0.2) !important;
}

/* Custom Message Card Styling */
.user-msg-box {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 16px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #f4f4f5;
}

.assistant-msg-box {
    background: #0f0f12;
    border: 1px solid #1f1f23;
    border-radius: 16px;
    padding: 16px 20px;
    margin: 8px 0;
    color: #e4e4e7;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

/* Tool execution visualizer card */
.tool-card {
    background: #121215;
    border-left: 3px solid #a855f7;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-family: 'Fira Code', monospace;
    font-size: 13px;
    color: #c084fc;
}

.tool-output {
    background: #0a0a0c;
    border-radius: 6px;
    padding: 8px 12px;
    margin-top: 6px;
    color: #e9d5ff;
    font-size: 12px;
    border: 1px solid #2e1065;
}

/* Starter Quick Cards */
.starter-card {
    background: #141417;
    border: 1px solid #27272a;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 12px;
    cursor: pointer;
    transition: all 0.2s ease-in-out;
}

.starter-card:hover {
    border-color: #52525b;
    background: #18181b;
    transform: translateY(-2px);
}

.starter-title {
    font-weight: 600;
    font-size: 14px;
    color: #f4f4f5;
    margin-bottom: 4px;
}

.starter-desc {
    font-size: 12px;
    color: #71717a;
}

/* Sidebar elements */
.stButton button {
    background-color: #18181b !important;
    color: #f4f4f5 !important;
    border: 1px solid #27272a !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stButton button:hover {
    background-color: #27272a !important;
    border-color: #3f3f46 !important;
    color: #ffffff !important;
}

/* Hide default hamburger & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(GROK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()

if "sessions_history" not in st.session_state:
    st.session_state.sessions_history = {
        st.session_state.thread_id: {
            "title": f"Session {st.session_state.thread_id}",
            "time": datetime.now().strftime("%I:%M %p"),
            "messages": []
        }
    }


# ---------------------------------------------------------------------------
# Sidebar - Grok Controls & Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; padding: 10px 0;'>
        <div style='font-size: 28px; font-weight: 900; color: #fff;'>𝕏</div>
        <div>
            <div style='font-weight: 700; font-size: 18px; color: #fff; letter-spacing: -0.5px;'>GROK CHATBOT</div>
            <div style='font-size: 11px; color: #a1a1aa;'>LangGraph + Groq Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # New Chat Session Button
    if st.button("➕ New Chat Session", use_container_width=True):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.thread_id = new_id
        st.session_state.messages = []
        st.session_state.sessions_history[new_id] = {
            "title": f"Session {new_id}",
            "time": datetime.now().strftime("%I:%M %p"),
            "messages": []
        }
        st.rerun()

    st.markdown("### 🤖 Model & Reasoning")

    # Groq Model Selector
    model_options = [
        "qwen/qwen3.6-27b",
        "groq/compound",
        "groq/compound-mini",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "Custom..."
    ]

    selected_model = st.selectbox(
        "Groq AI Model",
        options=model_options,
        index=0,
        help="Select the Groq high-speed LLM model architecture"
    )

    if selected_model == "Custom...":
        custom_model = st.text_input("Enter Model String", value="qwen/qwen3.6-27b")
        active_model_name = custom_model
    else:
        active_model_name = selected_model

    temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    st.markdown("### ⚙️ Integrated Tools")
    use_calc = st.checkbox("🧮 Math Calculator", value=True)
    use_hello = st.checkbox("👋 Adaptive Greetings", value=True)
    use_time = st.checkbox("⏰ System Clock Provider", value=True)

    # Active tool list assembly
    active_tools = []
    if use_calc:
        active_tools.append(calculator)
    if use_hello:
        active_tools.append(say_hello)
    if use_time:
        active_tools.append(get_system_time)

    st.markdown("### 💬 System Persona")
    system_persona = st.text_area(
        "System Prompt",
        value=(
            "You are Grok, an intelligent, ultra-fast AI assistant powered by Groq and LangGraph. "
            "You have access to real-time math, greeting, and system time tools. "
            "Provide witty, sharp, direct, concise, and helpful responses formatted in clean Markdown."
        ),
        height=100
    )

    st.markdown("---")
    st.markdown("### 📜 Session History")
    for s_id, s_data in list(st.session_state.sessions_history.items()):
        is_active = s_id == st.session_state.thread_id
        btn_label = f"{'🟢' if is_active else '💬'} {s_data['title']} ({s_data['time']})"
        if st.button(btn_label, key=f"session_{s_id}", use_container_width=True):
            st.session_state.thread_id = s_id
            st.session_state.messages = s_data["messages"]
            st.rerun()

    st.markdown("---")
    # Key Status Check
    env_key = os.getenv("GROQ_API_KEY", "")
    if not env_key and hasattr(st, "secrets"):
        try:
            if "GROQ_API_KEY" in st.secrets:
                env_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    key_input = st.text_input("Groq API Key (Optional Override)", type="password", value="")
    effective_key = key_input.strip() if key_input.strip() else env_key

    if effective_key:
        st.markdown("<span style='color: #22c55e; font-size: 12px;'>✔ API Key Active</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color: #ef4444; font-size: 12px;'>⚠️ GROQ_API_KEY Missing</span>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ReAct Agent Builder Helper
# ---------------------------------------------------------------------------
def get_agent_executor(model_name: str, temp: float, tools: list, persona: str, api_key: str):
    if not api_key:
        return None, "Groq API Key is not configured."
    
    try:
        model = ChatGroq(
            model=model_name,
            temperature=temp,
            groq_api_key=api_key
        )
        agent = create_react_agent(
            model=model,
            tools=tools,
            checkpointer=st.session_state.checkpointer,
            prompt=persona
        )
        return agent, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Main Header UI
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="grok-header">
    <div class="grok-logo">
        <span class="grok-logo-icon">𝕏 Grok</span>
        <span class="grok-badge">Groq ReAct Engine</span>
    </div>
    <div class="grok-status">
        <span class="pulse-dot"></span>
        <span>Model: <strong>{active_model_name}</strong> | Session: <code>{st.session_state.thread_id}</code></span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Quick Start Suggestion Cards (Show when history is empty)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("<h4 style='color: #a1a1aa; font-weight: 500;'>What can Grok help you with today?</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧮 Calculate Math Expressions\n*e.g. sqrt(144) + 10^2 * sin(pi/4)*", use_container_width=True):
            st.session_state.pending_prompt = "Calculate sqrt(144) + 10^2 * sin(pi/4) using the calculator tool."
            st.rerun()
            
        if st.button("⏰ Fetch System & Local Time\n*Retrieve high-precision system clock timestamps*", use_container_width=True):
            st.session_state.pending_prompt = "What is the current system time and UTC timestamp?"
            st.rerun()

    with col2:
        if st.button("👋 Personalized Friendly Greeting\n*Test adaptive tone greeting tool*", use_container_width=True):
            st.session_state.pending_prompt = "Greet me as 'User' in an energetic tone!"
            st.rerun()

        if st.button("🧠 Deep Logical Analysis\n*Solve complex multi-step reasoning problems*", use_container_width=True):
            st.session_state.pending_prompt = "If a train leaves Station A at 60 mph and another leaves Station B at 80 mph, explain step-by-step how to calculate their meeting point."
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render Chat History
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    tools_used = msg.get("tools_used", [])

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            # Display tool activity badges if any
            for tool_info in tools_used:
                st.markdown(f"""
                <div class="tool-card">
                    ⚡ <strong>Tool Invocation:</strong> <code>{tool_info['name']}</code>({tool_info['args']})
                    <div class="tool-output">⚙️ <strong>Result:</strong> {tool_info['result']}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(clean_response_text(content))


# ---------------------------------------------------------------------------
# Process Quick Prompt or User Input
# ---------------------------------------------------------------------------
prompt_to_process = None
if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    prompt_to_process = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

user_input = st.chat_input("Ask Grok anything... (e.g. math calculations, system time, greetings)")

if user_input:
    prompt_to_process = user_input

if prompt_to_process:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    st.session_state.sessions_history[st.session_state.thread_id]["messages"] = st.session_state.messages
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_to_process)

    # Initialize agent
    agent_executor, err_msg = get_agent_executor(
        model_name=active_model_name,
        temp=temperature,
        tools=active_tools,
        persona=system_persona,
        api_key=effective_key
    )

    with st.chat_message("assistant", avatar="🤖"):
        if err_msg:
            st.error(f"⚠️ Groq Agent Error: {err_msg}")
            st.info("Please set your `GROQ_API_KEY` in the `.env` file or sidebar input field.")
        else:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            assistant_response_placeholder = st.empty()
            tools_container = st.container()

            full_response = ""
            tools_recorded = []

            try:
                with st.spinner("Grok is thinking..."):
                    # Stream LangGraph ReAct agent updates
                    for chunk in agent_executor.stream(
                        {"messages": [HumanMessage(content=prompt_to_process)]},
                        config=config,
                        stream_mode="updates"
                    ):
                        # Tool execution output check
                        if "tools" in chunk and "messages" in chunk["tools"]:
                            for tool_msg in chunk["tools"]["messages"]:
                                tool_name = getattr(tool_msg, "name", "tool")
                                tool_res = tool_msg.content
                                tools_recorded.append({
                                    "name": tool_name,
                                    "args": "...",
                                    "result": tool_res
                                })
                                with tools_container:
                                    st.markdown(f"""
                                    <div class="tool-card">
                                        ⚙️ <strong>Tool Executed:</strong> <code>{tool_name}</code>
                                        <div class="tool-output">Output: {tool_res}</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                        # Agent response chunk check
                        if "agent" in chunk and "messages" in chunk["agent"]:
                            for agent_msg in chunk["agent"]["messages"]:
                                if hasattr(agent_msg, "tool_calls") and agent_msg.tool_calls:
                                    for tc in agent_msg.tool_calls:
                                        fn_name = tc.get("name", "tool")
                                        fn_args = ", ".join(f"{k}={v}" for k, v in tc.get("args", {}).items())
                                        with tools_container:
                                            st.markdown(f"""
                                            <div class="tool-card">
                                                ⚡ <strong>Calling Tool:</strong> <code>{fn_name}</code>({fn_args})
                                            </div>
                                            """, unsafe_allow_html=True)
                                elif agent_msg.content:
                                    full_response += agent_msg.content
                                    assistant_response_placeholder.markdown(clean_response_text(full_response) + "▌")

                assistant_response_placeholder.markdown(clean_response_text(full_response) if full_response else "*(Execution finished)*")

                # Save assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "tools_used": tools_recorded
                })
                st.session_state.sessions_history[st.session_state.thread_id]["messages"] = st.session_state.messages

                # Update session title based on first query
                if len(st.session_state.messages) == 2:
                    short_title = prompt_to_process[:24] + "..." if len(prompt_to_process) > 24 else prompt_to_process
                    st.session_state.sessions_history[st.session_state.thread_id]["title"] = short_title

            except Exception as ex:
                st.error(f"⚠️ Error executing query: {ex}")


if __name__ == "__main__":
    # Auto-launch Streamlit ONLY if executed directly via `python app.py` (not inside Streamlit runtime)
    if not st.runtime.exists():
        cmd = [sys.executable, "-m", "streamlit", "run", __file__]
        print(f"Launching Streamlit application via command: {' '.join(cmd)}")
        subprocess.run(cmd)


