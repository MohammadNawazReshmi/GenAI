import os
import sys
import ast
import math
import operator
import uuid
import warnings
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Suppress non-critical warnings for a clean terminal interface
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables from .env
load_dotenv()

# ANSI Color Codes for Rich Terminal Output
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
WHITE = "\033[97m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_CYAN = "\033[96m"
LIGHT_GRAY = "\033[37m"

# ---------------------------------------------------------------------------
# UI Theme Configuration (Change colors here)
# ---------------------------------------------------------------------------
USER_PROMPT_COLOR = f"{BOLD}{BLUE}"          # Color for "You > " label
USER_INPUT_COLOR = f"{CYAN}"                 # Color for text typed by the user
ASSISTANT_LABEL_COLOR = f"{BOLD}{GREEN}"     # Color for "Assistant > " label
ASSISTANT_TEXT_COLOR = f"{GREEN}"            # Color for Assistant output text
TOOL_CALL_COLOR = f"{MAGENTA}"               # Color for tool calling badge
TOOL_OUTPUT_COLOR = f"{YELLOW}"              # Color for tool execution result


# ---------------------------------------------------------------------------
# Safe Mathematical Expression Evaluator (AST-based, zero eval vulnerability)
# ---------------------------------------------------------------------------
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
}

SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node):
    """Recursively evaluates an AST node containing mathematical expressions."""
    if isinstance(node, ast.Constant):  # Python 3.8+ numbers/constants
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.Name):
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        raise ValueError(f"Unknown variable or constant: {node.id}")
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            func = SAFE_FUNCTIONS[node.func.id]
            args = [_safe_eval(arg) for arg in node.args]
            return func(*args)
        raise ValueError("Unsupported or unsafe function call.")
    else:
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Custom Agent Tools
# ---------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Useful for accurately evaluating mathematical calculations and formulas.
    Supports arithmetic (+, -, *, /, //, %, **), powers (^ converted to **),
    constants (pi, e), and functions (sqrt, sin, cos, tan, log, exp, abs, round, ceil, floor).
    Example inputs: '128 * 256', 'sqrt(144) + 10^2', 'sin(pi / 2)'
    """
    cleaned = expression.strip().replace("^", "**")
    try:
        parsed = ast.parse(cleaned, mode="eval")
        result = _safe_eval(parsed.body)
        return f"Calculation Result: {expression} = {result}"
    except Exception as exc:
        return f"Error evaluating expression '{expression}': {exc}"


@tool
def say_hello(name: str, tone: str = "friendly") -> str:
    """Useful for greeting a user by name with a specified tone (friendly, formal, energetic, or casual)."""
    greetings = {
        "formal": f"Good day, {name}. It is a pleasure to assist you today.",
        "energetic": f"Hey {name}! Super excited to collaborate with you! Let's get things done!",
        "casual": f"Hey there, {name}! How's it going today?",
        "friendly": f"Hello {name}! I hope you are having a wonderful and productive day.",
    }
    return greetings.get(tone.lower(), greetings["friendly"])


@tool
def get_system_time() -> str:
    """Useful for retrieving current local time, UTC timestamp, and date."""
    now = datetime.now()
    utc_now = datetime.utcnow()
    return (
        f"Local Time: {now.strftime('%A, %B %d, %Y - %I:%M:%S %p')} | "
        f"UTC Time: {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )


# ---------------------------------------------------------------------------
# Terminal UI Helpers
# ---------------------------------------------------------------------------
def print_banner(model_name: str, thread_id: str):
    """Renders a modern stylized ASCII banner for the assistant."""
    border = f"{CYAN}═" * 68 + f"{RESET}"
    print(f"\n{border}")
    print(f"{BOLD}{MAGENTA}   🚀 Python AI Chatbot {RESET}{DIM}(LangGraph ReAct + Groq LLM){RESET}")
    print(border)
    print(f" {BOLD}• Model:{RESET}    {GREEN}{model_name}{RESET}")
    print(f" {BOLD}• Session:{RESET}  {BLUE}{thread_id}{RESET}")
    print(f" {BOLD}• Memory:{RESET}   {YELLOW}Active (Multi-turn enabled){RESET}")
    print(f" {BOLD}• Commands:{RESET} {CYAN}help{RESET}, {CYAN}clear{RESET}, {CYAN}reset{RESET}, {CYAN}model{RESET}, {CYAN}quit{RESET}")
    print(border + "\n")


def print_help():
    """Displays user command guidance and tool info."""
    print(f"\n{BOLD}{CYAN}Available Commands:{RESET}")
    print(f"  {BOLD}help{RESET}   - Show this help menu")
    print(f"  {BOLD}clear{RESET}  - Clear the terminal screen")
    print(f"  {BOLD}reset{RESET}  - Start a new conversation session (clear memory)")
    print(f"  {BOLD}model{RESET}  - Display current model and connection status")
    print(f"  {BOLD}quit{RESET}   - Exit the chatbot (or type 'exit')\n")
    print(f"{BOLD}{CYAN}Integrated Tools:{RESET}")
    print(f"  {YELLOW}• calculator{RESET}      - Safe multi-function math evaluator (arithmetic, sqrt, trig, powers)")
    print(f"  {YELLOW}• say_hello{RESET}       - Adaptive greeting with customizable tone")
    print(f"  {YELLOW}• get_system_time{RESET} - Real-time local and UTC timestamp provider\n")


def clear_screen():
    """Clears the terminal console."""
    os.system("cls" if os.name == "nt" else "clear")


# ---------------------------------------------------------------------------
# Main Application Loop
# ---------------------------------------------------------------------------
def main():
    groq_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")

    if not groq_key:
        print(f"\n{YELLOW}{BOLD}⚠️  Warning:{RESET} GROQ_API_KEY is not set in environment or .env file.")
        print(f"Please add your key to {CYAN}.env{RESET}: {DIM}GROQ_API_KEY=gsk_...{RESET}\n")

    # Initialize Groq LLM
    model = ChatGroq(model=model_name, temperature=0)

    # Toolset registration
    tools = [calculator, say_hello, get_system_time]

    # Persistent in-memory checkpointer for multi-turn conversational context
    checkpointer = MemorySaver()

    # System instruction persona
    system_prompt = (
        "You are an intelligent, ultra-fast AI assistant powered by Groq and LangGraph. "
        "You have access to tools for mathematical calculations (calculator), greetings (say_hello), "
        "and real-time system timestamps (get_system_time). "
        "Use tools whenever precision or real-time info is required. "
        "Provide clear, concise, well-structured markdown answers."
    )

    # Compile ReAct agent with tools, stateful memory, and system prompt
    agent_executor = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        prompt=system_prompt,
    )

    # Session identifier
    thread_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": thread_id}}

    # Display startup banner
    print_banner(model_name, thread_id)

    while True:
        try:
            # Styled user prompt with custom color for both the label and typed input text
            user_input = input(f"\n{USER_PROMPT_COLOR}You > {RESET}{USER_INPUT_COLOR}").strip()
            print(f"{RESET}", end="")  # Reset formatting after input
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{YELLOW}Session ended. Goodbye! 👋{RESET}\n")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # Handle CLI commands
        if cmd in ("quit", "exit", "q"):
            print(f"\n{GREEN}Goodbye! Have a great day! 👋{RESET}\n")
            break
        elif cmd == "help":
            print_help()
            continue
        elif cmd in ("clear", "cls"):
            clear_screen()
            print_banner(model_name, thread_id)
            continue
        elif cmd in ("reset", "new"):
            thread_id = str(uuid.uuid4())[:8]
            config = {"configurable": {"thread_id": thread_id}}
            print(f"\n{GREEN}✔ Conversation memory reset.{RESET} New session ID: {BLUE}{thread_id}{RESET}\n")
            continue
        elif cmd == "model":
            has_key = "Set (valid format)" if (groq_key and groq_key.startswith("gsk_")) else "Configured" if groq_key else "Missing"
            print(f"\n{BOLD}• Model:{RESET} {GREEN}{model_name}{RESET} | {BOLD}API Key:{RESET} {has_key} | {BOLD}Session:{RESET} {BLUE}{thread_id}{RESET}\n")
            continue

        # Execute agent workflow with real-time streaming and tool activity badges
        print(f"\n{ASSISTANT_LABEL_COLOR}Assistant > {RESET}{ASSISTANT_TEXT_COLOR}", end="", flush=True)

        try:
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="updates",
            ):
                # Inspect tool calls and executions from graph updates
                if "tools" in chunk and "messages" in chunk["tools"]:
                    for tool_msg in chunk["tools"]["messages"]:
                        tool_name = getattr(tool_msg, "name", "tool")
                        print(f"\n{TOOL_OUTPUT_COLOR}⚙️  [{tool_name}]: {DIM}{tool_msg.content}{RESET}")
                        print(f"{ASSISTANT_LABEL_COLOR}Assistant > {RESET}{ASSISTANT_TEXT_COLOR}", end="", flush=True)

                if "agent" in chunk and "messages" in chunk["agent"]:
                    for agent_msg in chunk["agent"]["messages"]:
                        # If agent decides to call a tool, display informative badge
                        if hasattr(agent_msg, "tool_calls") and agent_msg.tool_calls:
                            for tc in agent_msg.tool_calls:
                                fn_name = tc.get("name", "tool")
                                args_str = ", ".join(f"{k}={repr(v)}" for k, v in tc.get("args", {}).items())
                                print(f"\n{TOOL_CALL_COLOR}⚡ Calling Tool: {BOLD}{fn_name}({args_str}){RESET}", flush=True)
                        elif agent_msg.content:
                            print(f"{ASSISTANT_TEXT_COLOR}{agent_msg.content}", end="", flush=True)

            print(f"{RESET}")  # Reset formatting and add newline after response

        except Exception as err:
            print(f"\n{RED}{BOLD}⚠️ Error processing request:{RESET} {err}")
            print(f"{DIM}Tip: Check your GROQ_API_KEY in .env or verify network connectivity.{RESET}\n")


if __name__ == "__main__":
    main()