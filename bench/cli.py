"""Command line entry point: `python -m bench <command>`."""

from __future__ import annotations

import argparse

import litellm
from dotenv import load_dotenv

from bench import metrics
from bench.models import call_model, load_config


def cmd_hello(args: argparse.Namespace) -> int:
    config = load_config()
    model = args.model or config["models"][0]["id"]
    settings = config.get("settings", {})
    try:
        result = call_model(
            model,
            args.prompt,
            temperature=settings.get("temperature", 0.2),
            max_tokens=settings.get("max_tokens", 1024),
            mock_response="Hello from a mocked model." if args.mock else None,
        )
    except litellm.AuthenticationError:
        print(f"No working API key for {model}. Add it to .env (see .env.example), or try --mock.")
        return 1
    cost = "unknown (no price listed)" if result.cost_usd is None else f"${result.cost_usd:.6f}"
    print(result.text.strip())
    print()
    print(f"model:   {result.model}{' (mocked)' if args.mock else ''}")
    print(f"tokens:  {result.input_tokens} in / {result.output_tokens} out")
    print(f"cost:    {cost}")
    print(f"latency: {result.latency_s:.2f}s")
    return 0


def cmd_lock(args: argparse.Namespace) -> int:
    try:
        record = metrics.lock(args.task)
    except ValueError as e:
        print(f"Not locked: {e}")
        return 1
    print(f"Locked {args.task} metrics at {record['locked_at']}")
    print(f"sha256: {record['sha256']}")
    if record["status"] != "reviewed":
        print("Note: status is not 'reviewed' yet; have the PMM sign off before real runs.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    ok = metrics.verify(args.task)
    print("Metrics match their lock." if ok else "Metrics changed since locking (or were never locked).")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    hello = sub.add_parser("hello", help="Call one model and print its answer and cost.")
    hello.add_argument("--model", help="LiteLLM model id; defaults to the first model in models.yaml.")
    hello.add_argument("--prompt", default="In one sentence, what does a product marketing manager do?")
    hello.add_argument("--mock", action="store_true", help="Skip the network and the API key; returns canned text.")
    hello.set_defaults(func=cmd_hello)

    lock = sub.add_parser("lock", help="Validate a task's metrics.yaml and record its hash.")
    lock.add_argument("task")
    lock.set_defaults(func=cmd_lock)

    verify = sub.add_parser("verify", help="Check a task's metrics haven't changed since locking.")
    verify.add_argument("task")
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)
