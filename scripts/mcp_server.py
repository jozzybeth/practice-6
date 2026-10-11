import json
import logging
import time
from pathlib import Path
import sys
import asyncio
import yaml
from langfuse import get_client
from langchain_groq import ChatGroq
from mcp.server.fastmcp import FastMCP


logging.getLogger("mcp").setLevel(logging.ERROR)
logging.getLogger("mcp.server").setLevel(logging.ERROR)
logging.getLogger("mcp.server.lowlevel").setLevel(logging.ERROR)


def load_config():
    """Load the project configuration."""
    ROOT_DIR = Path(__file__).resolve().parent.parent
    config_path = ROOT_DIR / "configs" / "agent.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_articles(input_path: Path) -> list[dict]:
    """Load articles from the input JSON file."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        articles = json.load(file)

    if not isinstance(articles, list):
        raise ValueError("Input JSON must contain a list of articles.")

    return articles


def create_mcp_server(config: dict) -> FastMCP:
    """Create an MCP server and its tools using the supplied configuration."""
    data_config = config["data"]
    model_config = config["model"]

    ROOT_DIR = Path(__file__).resolve().parent.parent
    input_path = (ROOT_DIR / data_config["input_path"]).resolve()
    output_path = (ROOT_DIR / data_config["output_path"]).resolve()

    articles = {
        str(article["id"]): article
        for article in load_articles(input_path)
    }

    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")

    llm = ChatGroq(
        model=model_config["name"],
        temperature=model_config.get("temperature", 0),
        max_tokens=512,
        max_retries=15,
    )

    langfuse_client = get_client()
    mcp = FastMCP("article-summarizer", log_level="ERROR")

    @mcp.tool()
    def reset_results() -> str:
        """Clear the output file before processing articles."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

        return "Output file initialized."

    @mcp.tool()
    def list_articles() -> str:
        """Return IDs of all articles."""
        return json.dumps(list(articles.keys()), ensure_ascii=False)

    @mcp.tool()
    async def process_article(article_id: str) -> str:
        """Summarize an article by ID and save its summary."""
        await asyncio.sleep(15.0)

        article = articles[article_id]
        title = str(article["title"])
        article_text = str(article["text"])

        prompt = config["prompts"]["summarization"].format(
            title=title,
            article_text=article_text,
        )

        with langfuse_client.start_as_current_observation(
            as_type="generation",
            name="article_summary_llm",
            model=model_config["name"],
            input=prompt,
        ) as generation:
            response = await llm.ainvoke(prompt)
            summary = response.content.strip()
            generation.update(output=summary)

        langfuse_client.flush()

        if not summary:
            raise ValueError("Generated summary is empty.")

        result_item = {
            "id": str(article_id),
            "title": title,
            "summary": summary,
        }

        with output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result_item, ensure_ascii=False) + "\n")

        return (
            f"Successfully processed and saved summary for article "
            f"'{title}' (ID: {article_id})."
        )

    return mcp


if __name__ == "__main__":
    server = create_mcp_server(load_config())
    server.run(transport="stdio")
