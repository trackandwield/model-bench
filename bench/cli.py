"""Command line entry point: `python -m bench <command>`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import litellm
from dotenv import load_dotenv

from bench import checks, compare, dataset, judge, metrics, plan, tasks, workflow
from bench.models import call_model, load_config
from bench.store import Store


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
    budget = args.budget if args.budget is not None else load_config().get("budget_usd")
    run_id, repeats, mock = None, args.repeats, args.mock
    if args.resume:
        # Same run id, models, cases, repeats and settings as the saved run. Finished
        # calls come from the cache, so only what's missing is paid for.
        config = json.loads(Path(args.resume).read_text())
        args.task, args.case, args.limit = config["task"], config["case_ids"], None
        models, settings, repeats, mock = config["models"], config["settings"], config["repeats"], config["mock"]
        run_id = config["run_id"]
        if args.budget is None:
            budget = config.get("budget_usd")
        changed = plan.changed_since(config)
        if changed:
            print(f"Warning: {', '.join(changed)} changed since run {run_id}; results won't match the original.")

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

    store = Store()
    print(f"{task.name}: {task.kind} with {len(task.steps)} step(s) × {len(cases)} case(s) × "
          f"{len(models)} model(s) × {repeats} repeat(s)")
    est = plan.estimate(task, cases, models, repeats=repeats, settings=settings, store=store, prices=prices,
                        mock=mock, use_cache=not args.no_cache)
    _print_estimate(est, models, budget)
    if args.estimate:
        return 0
    if budget is not None and est.cost_usd > budget:
        print(f"\nNot run: the estimate is over the ${budget:.2f} budget. "
              f"Rerun with --budget {est.cost_usd * 1.2:.2f} (or more) to go ahead.")
        return 1

    run_id = run_id or compare.new_run_id()
    config = plan.build_config(run_id, task.name, models, [c["id"] for c in cases], repeats=repeats,
                               settings=settings, budget_usd=budget, mock=mock)
    if not config["metrics_locked"] and not mock:
        print("Note: metrics aren't locked yet (bench lock), so this run can't claim pre-registered metrics.")
    # A resume keeps the original config, so it still says what the run first ran against.
    config_file = Path(args.resume) if args.resume else plan.save_config(config)
    cap = workflow.Budget(budget)
    specs = checks.check_specs(metrics.load(args.task))
    run_id, results, path = workflow.run(task, cases, models, settings=settings, mock=mock,
                                         use_cache=not args.no_cache, prices=prices, check_specs=specs,
                                         repeats=repeats, budget=cap, run_id=run_id, store=store)

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
    _print_checks(models, results, [s["name"] for s in specs])
    print(f"\nrun {run_id}: spent ${cap.spent:.4f}; outputs, checks and every step's prompt in {path}")
    unfinished = sum(1 for r in results if r.failed_step)
    if unfinished:
        print(f"{unfinished} chain(s) didn't finish. To retry only those (finished calls are free):")
        print(f'  uv run python -m bench run --resume "{config_file}"')
    else:
        print(f"config (to reproduce): {config_file}")
    return 1 if all(r.failed_step for r in results) else 0


def _print_estimate(est, models: list[str], budget: float | None) -> None:
    print(f"\nestimate: {est.calls} calls, {est.cached} already cached (free), {est.to_make} to make")
    for m in models:
        cost = "unknown price" if m in est.unpriced else f"${est.by_model.get(m, 0.0):.4f}"
        guess = "  (no history yet, assumed a long answer)" if m in est.guessed and m not in est.unpriced else ""
        print(f"  {m}: {cost}{guess}")
    known = f"${est.cost_usd:.4f}" + (" plus unpriced models" if est.unpriced else "")
    print(f"  total: about {known}" + (f"; budget ${budget:.2f}" if budget is not None else ""))


def _print_checks(models: list[str], results, names: list[str]) -> None:
    """Pass counts per model per check, then the first few failures as evidence."""
    if not names:
        return
    width = max(len(n) for n in names)
    short = {m: m.split("/")[-1][:18] for m in models}
    print(f"\n{'check (passed/checked)':<{width}}  " + "  ".join(f"{short[m]:>18}" for m in models))
    for n in names:
        cells = []
        for m in models:
            checked = [r.checks[n] for r in results if r.model == m and n in r.checks]
            cells.append(f"{sum(c.passed for c in checked)}/{len(checked)}" if checked else "-")
        print(f"{n:<{width}}  " + "  ".join(f"{c:>18}" for c in cells))
    shown = 0
    for r in results:
        for n, c in r.checks.items():
            if not c.passed and c.detail and shown < 5:
                print(f"  {r.model} {r.case_id} {n}: {', '.join(map(str, c.detail))}")
                shown += 1


def cmd_check(args: argparse.Namespace) -> int:
    """Re-run the automatic checks on an earlier run's file. No model calls, no cost."""
    path = Path(args.file)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    task_name = rows[0]["task"]
    specs = checks.check_specs(metrics.load(task_name))
    cases = {c["id"]: c for c in dataset.load(task_name)}
    results = []
    for row in rows:
        r = workflow.CaseResult(row["model"], row["case_id"])
        if row.get("output") is not None:
            r.checks = checks.run_checks(row["output"], cases[row["case_id"]]["inputs"], specs)
        row["checks"] = {k: v.to_dict() for k, v in r.checks.items()}
        results.append(r)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    models = list(dict.fromkeys(row["model"] for row in rows))
    _print_checks(models, results, [s["name"] for s in specs])
    print(f"\nchecks saved into {path}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    """Score an earlier run's outputs with the LLM judge and save the scores into its file."""
    path = Path(args.file)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    task_name = rows[0]["task"]
    spec = metrics.load(task_name)
    judge_specs = judge.judge_metrics(spec)
    if not judge_specs:
        print(f"{task_name} has no judge metrics.")
        return 1
    config = load_config()
    judges = args.judge or config.get("judges") or []
    settings = {k: (config.get("judge_settings") or {}).get(k, d) for k, d in (("temperature", 0), ("max_tokens", 4096))}
    cases = {c["id"]: c for c in dataset.load(task_name)}
    budget = args.budget if args.budget is not None else config.get("budget_usd")
    todo = sum(1 for r in rows if r.get("output") is not None)
    if args.sample is not None:
        todo = min(todo, args.sample)
    print(f"Judging {todo} output(s) × {len(judge_specs)} metric(s) with {', '.join(judges)}"
          f"{' (mocked)' if args.mock else ''}; {args.repeat} repeat(s) for consistency")
    store = Store()
    opts = dict(settings=settings, store=store, mock=args.mock, use_cache=not args.no_cache,
                repeat_sample=args.repeat, sample=args.sample)
    est = judge.estimate(rows, judge_specs, judges, cases, **opts)
    _print_estimate(est, judges, budget)
    if args.estimate:
        return 0
    if budget is not None and est.cost_usd > budget:
        print(f"\nNot judged: the estimate is over the ${budget:.2f} budget. Rerun with --budget "
              f"{est.cost_usd * 1.2:.2f} (or more), or judge a sample with --sample N.")
        return 1
    cap = workflow.Budget(budget)
    run_id, first, repeats = judge.run(rows, spec, judges, cases, budget=cap, **opts)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    names = [m["name"] for m in judge_specs]
    models = list(dict.fromkeys(r["model"] for r in rows))
    short = {m: m.split("/")[-1][:18] for m in models}
    width = max(len(n) for n in names + ["judge metric"])
    print(f"\n{'avg score (passed/judged)':<{width}}  " + "  ".join(f"{short[m]:>18}" for m in models))
    for n in names:
        cells = []
        for m in models:
            js = [r["judge"][n] for r in rows if r["model"] == m and n in r.get("judge", {})]
            scored = [j for j in js if j["score"] is not None]
            if not scored:
                cells.append("-")
                continue
            avg = sum(j["score"] for j in scored) / len(scored)
            cells.append(f"{avg:.1f} ({sum(bool(j['passed']) for j in scored)}/{len(scored)})")
        print(f"{n:<{width}}  " + "  ".join(f"{c:>18}" for c in cells))

    errors = [(rows[i], j) for i, j in first if j.error]
    for row, j in errors[:3]:
        print(f"\n{row['model']} {row['case_id']} {j.metric}: judge failed: {j.error}")
    worst = sorted(((rows[i], j) for i, j in first if j.score is not None), key=lambda x: x[1].score)[:3]
    if worst:
        print("\nLowest scores, with the judge's reasoning:")
    for row, j in worst:
        print(f"  {row['model']} {row['case_id']} {j.metric} = {j.score}: {j.reasoning}")
        for q in j.evidence[:2]:
            print(f"    quoted: \"{q}\"")
    unquoted = sum(1 for _, j in first if j.missing_quotes)
    if unquoted:
        print(f"\n{unquoted} judgment(s) quoted text that isn't in the output; check those by hand.")

    c = judge.consistency(first, repeats)
    if c:
        print(f"\nConsistency: judged {c['pairs']} again; same score {c['exact']:.0%}, within 1 point {c['within_1']:.0%}")
    stopped = sum(1 for _, j in first + repeats if j.error and j.error.startswith("Budget"))
    if stopped:
        print(f"\n{stopped} judgment(s) skipped at the budget cap. Rerun the same command to finish; "
              "finished judgments are free.")
    print(f"\njudge run {run_id}: spent ${cap.spent:.4f}; scores saved into {path}")
    return 1 if first and all(j.error for _, j in first) else 0


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
    run.add_argument("task", nargs="?", help="Folder under tasks/, e.g. battlecard")
    run.add_argument("--case", action="append", help="Only this case id (repeatable).")
    run.add_argument("--limit", type=int, help="Only the first N cases.")
    run.add_argument("--model", action="append", help="Limit to this model id (repeatable).")
    run.add_argument("--mock", action="store_true", help="Skip the network; canned answers.")
    run.add_argument("--no-cache", action="store_true", help="Call the models even if an identical call is cached.")
    run.add_argument("--repeats", type=int, default=1, help="Run each model on each case this many times.")
    run.add_argument("--budget", type=float, help="Spend cap in USD; defaults to budget_usd in models.yaml.")
    run.add_argument("--estimate", action="store_true", help="Print the cost estimate and stop.")
    run.add_argument("--resume", metavar="CONFIG", help="Finish or reproduce a run from its runs/<id>-<task>.config.json.")
    run.set_defaults(func=cmd_run)

    chk = sub.add_parser("check", help="Re-run the automatic checks on an earlier run's .jsonl file (free).")
    chk.add_argument("file")
    chk.set_defaults(func=cmd_check)

    jdg = sub.add_parser("judge", help="Score an earlier run's outputs with the LLM judge.")
    jdg.add_argument("file", help="A run file, runs/<run id>-<task>.jsonl")
    jdg.add_argument("--judge", action="append", help="Judge model id (repeatable); defaults to `judges` in models.yaml.")
    jdg.add_argument("--repeat", type=int, default=5, help="Judge this many random outputs twice to check consistency.")
    jdg.add_argument("--sample", type=int, help="Judge only this many random outputs.")
    jdg.add_argument("--budget", type=float, help="Spend cap in USD; defaults to budget_usd in models.yaml.")
    jdg.add_argument("--estimate", action="store_true", help="Print the cost estimate and stop.")
    jdg.add_argument("--mock", action="store_true", help="Skip the network; canned verdicts.")
    jdg.add_argument("--no-cache", action="store_true", help="Call the judge even if an identical call is cached.")
    jdg.set_defaults(func=cmd_judge)

    lock = sub.add_parser("lock", help="Validate a task's metrics.yaml and record its hash.")
    lock.add_argument("task")
    lock.set_defaults(func=cmd_lock)

    verify = sub.add_parser("verify", help="Check a task's metrics haven't changed since locking.")
    verify.add_argument("task")
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and not (args.task or args.resume):
        parser.error("run needs a task, or --resume CONFIG")
    return args.func(args)
