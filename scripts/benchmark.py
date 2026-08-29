#!/usr/bin/env python3
"""Run a small latency experiment against the Ollama HTTP API."""

import argparse
import json
import time
import urllib.error
import urllib.request


def run(model: str, prompt: str, url: str) -> int:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"Request failed: {error}")
        return 1

    elapsed = time.perf_counter() - started
    print(f"model: {model}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print("response:")
    print(result.get("response", ""))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Ollama model tag")
    parser.add_argument("--prompt", required=True, help="Synthetic benchmark prompt")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL (default: http://127.0.0.1:11434)",
    )
    args = parser.parse_args()
    return run(args.model, args.prompt, args.url)


if __name__ == "__main__":
    raise SystemExit(main())
