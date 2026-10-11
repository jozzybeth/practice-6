import json
import yaml
from pathlib import Path
from mcp_server import load_config


def load_results(result_path) -> list[dict]:
    """Load summaries from a JSON file."""
    if not result_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {result_path}"
        )

    with result_path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    config = load_config()
    ROOT_DIR = Path(__file__).resolve().parent.parent
    result_path = ROOT_DIR / config["data"]["output_path"]
    results = load_results(result_path)

    if not results:
        print("No results yet.")
        return

    for item in results:
        print(f"\n{'=' * 80}")
        print(f"ID: {item['id']}")
        print(f"Title: {item['title']}")
        print(f"\nSummary:\n{item['summary']}")


if __name__ == "__main__":
    main()