import asyncio
import logging
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.langchain import CallbackHandler

from agent_graph import create_agent
from mcp_server import load_config


logging.basicConfig(level=logging.ERROR)
for logger_name in ["mcp", "httpx", "langchain", "langgraph", "asyncio"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


async def main():
    config = load_config()
    agent = await create_agent(config)

    input_path = config["data"]["input_path"]
    output_path = config["data"]["output_path"]
    recursion_limit = config["agent"].get("recursion_limit", 500)

    langfuse_handler = CallbackHandler()

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Model: {config['model']['name']}")
    print("\nAgent started...\n" + "=" * 60)

    async for chunk in agent.astream(
        {
            "messages": [
                SystemMessage(content=config["prompts"]["agent"]),
                HumanMessage(content=config["prompts"]["task"]),
            ]
        },
        config={
            "recursion_limit": recursion_limit,
            "callbacks": [langfuse_handler],
        },
        stream_mode="updates",
    ):
        for node_name, state_update in chunk.items():
            if node_name == "llm":
                last_msg = state_update["messages"][-1]

                if getattr(last_msg, "tool_calls", None):
                    for tool_call in last_msg.tool_calls:
                        print(f"\n[LLM]: Call tool -> {tool_call['name']}")
                        print(f"   Arguments: {tool_call['args']}")
                else:
                    print(f"\n[LLM Final answer]:\n{last_msg.content}")

            elif node_name == "tools":
                for tool_msg in state_update["messages"]:
                    print(f"[Tool '{tool_msg.name}']: {tool_msg.content}")

    print("\n" + "=" * 60 + "\nMission completed!")


if __name__ == "__main__":
    asyncio.run(main())