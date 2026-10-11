import os
import sys
from pathlib import Path

from langchain_core.messages import trim_messages
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import (
    END,
    START,
    MessagesState,
    StateGraph,
)
from langgraph.prebuilt import ToolNode


def should_continue(state: MessagesState) -> str:
    """Route tool calls to the tools node."""
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "end"


async def create_agent(config: dict):
    """Create a LangGraph agent using the supplied configuration."""
    ROOT_DIR = Path(__file__).resolve().parent.parent
    client = MultiServerMCPClient(
        {
            "article_tools": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(ROOT_DIR / "scripts" / "mcp_server.py")],
                "env": dict(os.environ),
            }
        }
    )

    tools = await client.get_tools()
    model_config = config["model"]

    llm = ChatGroq(
        model=model_config["name"],
        temperature=model_config.get("temperature", 0),
        max_tokens=512,
        max_retries=15,
    )

    llm_with_tools = llm.bind_tools(tools)

    async def call_model(state: MessagesState):
        trimmed_messages = trim_messages(
            state["messages"],
            max_tokens=1500,
            strategy="last",
            token_counter=len,
            start_on="human",
            include_system=True,
        )

        response = await llm_with_tools.ainvoke(trimmed_messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)

    builder.add_node("llm", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "llm")

    builder.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    builder.add_edge("tools", "llm")

    return builder.compile()
