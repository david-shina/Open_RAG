import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from tests.integration.real_llm_cases import (  # noqa: E402
    MODEL,
    REQUEST_TIMEOUT,
    TRANSLATOR_MODEL,
    run_all_cases,
    summarize,
)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real-Groq smoke suite and record results.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of cases to run.")
    parser.add_argument(
        "--output",
        default="docs/real_llm_smoke.json",
        help="Path for the JSON artifact.",
    )
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not configured. Add it to .env or the environment.", file=sys.stderr)
        return 2

    results = run_all_cases(limit=args.limit)
    summary = summarize(results)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "model": MODEL,
        "translator_model": TRANSLATOR_MODEL,
        "temperature": 0,
        "request_timeout_seconds": REQUEST_TIMEOUT,
        "cases": [asdict(result) for result in results],
        "summary": summary,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("| Case | Passed | Latency (ms) | Tokens | Calls |")
    print("|---|---|---|---|---|")
    for result in results:
        print(
            f"| {result.name} | {result.passed} | {result.latency_ms:.2f} | "
            f"{result.usage.get('total_tokens', 0)} | {result.usage.get('calls', 0)} |"
        )
    print(
        f"\n{summary['passed']}/{summary['total']} passed, "
        f"{summary['total_tokens']} tokens, "
        f"p50 latency {summary['p50_latency_ms']} ms"
    )
    print(f"Wrote {output_path}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
