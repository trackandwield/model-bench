"""Command line entry point: `python -m bench <command>`."""

from __future__ import annotations

import argparse

import litellm
from dotenv import load_dotenv

from bench import compare, dataset, metrics, tasks, workflow
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


def _money(v: float | None) -> str:
    return "unknown" if v is None else f"${v:.6f}"


def _run_config(args: argparse.Namespace) -> tuple[list[str], dict, dict]:
    """Models, shared settings and configured prices from models.yaml (and --model)."""
    config = load_config()
    models = args.model or [m["id"] for m in config["models"]]
    settings = {k: config.get("settings", {}).get(k, d) for k, d in (("temperature", 0.2), ("max_tokens", 1024))}
    prices = {
        m["id"]: m["price_per_mtok"]
        for m in config["models"]
        if all(isinstance((m.get("price_per_mtok") or {}).get(k), (int, float)) for k in ("input", "output"))
    }
    return models, settings, prices


def cmd_compare(args: argparse.Namespace) -> int:
    models, settings, prices = _run_config(args)
    run_id, results, path = compare.run(
        models, args.prompt, system=args.system, settings=settings, mock=args.mock,
        use_cache=not args.no_cache, prices=prices,
    )

    width = max(len(m) for m in models)
    print(f"{'model':<{width}}  {'tokens in/out':>13}  {'cost':>11}  {'latency':>8}  status")
    for r in results:
        status = "error" if r.error else "refused" if r.refused else "cached" if r.cached else "ok"
        tokens = f"{r.input_tokens}/{r.output_tokens}"
        print(f"{r.model:<{width}}  {tokens:>13}  {_money(r.cost_usd):>11}  {r.latency_s:>7.2f}s  {status}")
    for r in results:
        if r.error:
            print(f"\n{r.model} failed: {r.error}")
    known = [r.cost_usd for r in results if r.cost_usd is not None]
    print(f"\nrun {run_id}: total cost ${sum(known):.6f}; outputs in {path}")
    return 1 if all(r.error for r in results) else 0


def cmd_run(args: argparse.Namespace) -> int:
    models, settings, prices = _run_config(args)
    task = tasks.load(args.task)
    cases = dataset.load(args.task)
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cases}
        if missing:
            print(f"Unknown case ids: {', '.join(sorted(missing))}")
            return 1
    if args.limit:
        cases = cases[: args.limit]

    print(f"{task.name}: {task.kind} with {len(task.steps)} step(s) × {len(cases)} case(s) × {len(models)} model(s)")
    run_id, results, path = workflow.run(task, cases, models, settings=settings, mock=args.mock,
                                         use_cache=not args.no_cache, prices=prices)

    width = max(len(m) for m in models)
    print(f"\n{'model':<{width}}  {'ok':>7}  {'cost':>11}  {'avg latency':>11}  failed at")
    for m in models:
        rows = [r for r in results if r.model == m]
        ok = sum(1 for r in rows if not r.failed_step)
        costs = [r.cost_usd for r in rows if r.cost_usd is not None]
        cost = _money(sum(costs)) if len(costs) == len(rows) else "unknown"
        avg = sum(r.latency_s for r in rows) / len(rows)
        fails = sorted({r.failed_step for r in rows if r.failed_step})
        print(f"{m:<{width}}  {ok:>3}/{len(rows):<3}  {cost:>11}  {avg:>10.2f}s  {', '.join(fails) or '-'}")
    for r in results:
        if r.failed_step:
            err = next(s.error for s in r.steps if s.error)
            print(f"\n{r.model} on {r.case_id} failed at {r.failed_step}: {err}")
            break
    print(f"\nrun {run_id}: outputs and every step's prompt in {path}")
    return 1 if all(r.failed_step for r in results) else 0


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

    cmp = sub.add_parser("compare", help="Run one prompt on every model in models.yaml.")
    cmp.add_argument("--prompt", required=True)
    cmp.add_argument("--system", help="Optional system prompt, the same for every model.")
    cmp.add_argument("--model", action="append", help="Limit to this model id (repeatable).")
    cmp.add_argument("--mock", action="store_true", help="Skip the network; canned answers per model.")
    cmp.add_argument("--no-cache", action="store_true", help="Call the models even if an identical call is cached.")
    cmp.set_defaults(func=cmd_compare)

    run = sub.add_parser("run", help="Run a task on every model × its test cases.")
    run.add_argument("task", help="Folder under tasks/, e.g. battlecard")
    run.add_argument("--case", action="append", help="Only this case id (repeatable).")
    run.add_argument("--limit", type=int, help="Only the first N cases.")
    run.add_argument("--model", action="append", help="Limit to this model id (repeatable).")
    run.add_argument("--mock", action="store_true", help="Skip the network; canned answers.")
    run.add_argument("--no-cache", action="store_true", help="Call the models even if an identical call is cached.")
    run.set_defaults(func=cmd_run)

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
