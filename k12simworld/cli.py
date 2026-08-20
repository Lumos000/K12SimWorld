"""Command-line interface for curation, generation, validation, and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from .adapters import VisPhyLLMAdapter, VisPhyRendererAdapter
from .baselines import BASELINE_METHODS, BaselinePipeline
from .curation import assign_knowledge_disjoint_splits, build_benchmark
from .domain_compiler import render_domain_html
from .domain_solvers import DOMAIN_ENGINES, domain_entity_ids, simulate_domain
from .evaluation.report import write_evaluation_report
from .evaluation.traces import score_trace
from .human_selection import prepare_human_physics_selection
from .io import read_records, safe_artifact_name, write_json, write_jsonl
from .models import EduWorldSpec, K12Problem
from .models import RenderSpec
from .pipeline import K12SimWorldPipeline
from .prompts import storyboard_prompt
from .tiering import partition_physics_tiers
from .validation import validate_program_payload


def _problems(path: str) -> List[K12Problem]:
    return [K12Problem.from_record(record) for record in read_records(path)]


def command_curate(args: argparse.Namespace) -> int:
    result = build_benchmark(
        _problems(args.input),
        physics_target=args.physics_target,
        extension_target=args.extension_target,
        expert_target=args.expert_target,
        seed=args.seed,
    )
    target = Path(args.output_dir)
    splits = assign_knowledge_disjoint_splits(result.selected, seed=args.seed)
    benchmark = []
    for problem in result.selected:
        record = problem.benchmark_record()
        record["split"] = splits[problem.problem_id]
        record["expert_reference"] = problem.problem_id in set(result.expert_subset_ids)
        benchmark.append(record)
    write_jsonl(target / "k12simbench.jsonl", benchmark)
    write_json(target / "expert_subset_ids.json", result.expert_subset_ids)
    write_jsonl(
        target / "rejected.jsonl",
        ({**problem.benchmark_record(), "selection": decision.__dict__} for problem, decision in result.rejected),
    )
    write_json(target / "curation_summary.json", result.summary)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    target = Path(args.output_dir)
    for problem in _problems(args.benchmark)[: args.limit or None]:
        path = target / f"{problem.problem_id}.storyboard.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(storyboard_prompt(problem), encoding="utf-8")
    return 0


def command_prepare_human_selection(args: argparse.Namespace) -> int:
    report = prepare_human_physics_selection(
        args.input,
        args.selection,
        args.output_dir,
        screening_path=args.screening_results,
        smoke_target=args.smoke_target,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    has_integrity_error = bool(
        report["missing_ids"] or report["nonphysics_ids"] or report["raw_duplicate_ids"]
    )
    return 1 if has_integrity_error and not args.allow_integrity_errors else 0


def command_partition_physics_tiers(args: argparse.Namespace) -> int:
    report = partition_physics_tiers(args.benchmark, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_simulate_domain(args: argparse.Namespace) -> int:
    source = Path(args.spec)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("domain spec must be a JSON object")
    trace = simulate_domain(args.engine, value)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    object_ids = sorted(domain_entity_ids(args.engine, value))
    document = render_domain_html(args.engine, value, trace, object_ids)
    trace_path = output / "simulation_trace.json"
    html_path = output / "scene.html"
    write_json(trace_path, trace)
    html_path.write_text(document, encoding="utf-8")
    videos: List[str] = []
    if args.render:
        playback = float(value.get("playback_duration") or value.get("duration") or 8.0)
        playback = min(30.0, max(1.0, playback))
        adapter = VisPhyRendererAdapter(
            args.engine,
            str(output / "renderer"),
            RenderSpec(engine=args.engine, duration=playback),
        )
        videos.append(adapter.render(document, str(output / "scene.mp4")))
    summary = {
        "engine": args.engine,
        "solver_trace": str(trace_path),
        "html": str(html_path),
        "videos": videos,
        "spec_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "trace_summary": trace.get("summary") or {},
    }
    write_json(output / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_generate(args: argparse.Namespace) -> int:
    def pipeline_factory() -> K12SimWorldPipeline:
        return K12SimWorldPipeline(
            VisPhyLLMAdapter(args.model), args.output_dir, render=args.render
        )

    return _run_generation(args, pipeline_factory(), pipeline_factory=pipeline_factory)


def command_generate_baseline(args: argparse.Namespace) -> int:
    def pipeline_factory() -> BaselinePipeline:
        return BaselinePipeline(
            VisPhyLLMAdapter(args.model),
            args.output_dir,
            method=args.method,
            render=args.render,
        )

    return _run_generation(args, pipeline_factory(), pipeline_factory=pipeline_factory)


def _run_generation(
    args: argparse.Namespace, pipeline: Any, *, pipeline_factory: Any = None
) -> int:
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    jobs = int(getattr(args, "jobs", 1) or 1)
    if jobs < 1:
        raise ValueError("--jobs must be at least 1")
    if jobs > 1 and pipeline_factory is None:
        raise ValueError("parallel generation requires a pipeline factory")

    output = Path(args.output_dir)
    problems = _problems(args.benchmark)[: args.limit or None]
    records: List[Dict[str, Any] | None] = [None] * len(problems)
    pending: List[tuple[int, K12Problem]] = []

    for index, problem in enumerate(problems):
        item_manifest = output / safe_artifact_name(problem.problem_id) / "manifest.json"
        existing: Dict[str, Any] | None = None
        if args.resume and item_manifest.is_file():
            try:
                value = json.loads(item_manifest.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    existing = value
            except (OSError, json.JSONDecodeError):
                existing = None
        should_skip = existing is not None and (
            bool(existing.get("success")) or not args.retry_failed
        )
        if should_skip:
            records[index] = existing
            print(
                f"{problem.problem_id}: skipped existing "
                f"{'success' if existing.get('success') else 'failure'}"
            )
        else:
            pending.append((index, problem))

    def persist() -> None:
        # Completed records stay in benchmark order; item manifests remain authoritative.
        write_jsonl(
            output / "manifests.jsonl",
            [record for record in records if record is not None],
        )

    def run_one(worker: Any, index: int, problem: K12Problem) -> tuple[int, Dict[str, Any]]:
        manifest = worker.generate(problem, requested_engine=args.engine)
        return index, manifest.to_dict()

    if jobs == 1:
        for index, problem in pending:
            index, record = run_one(pipeline, index, problem)
            records[index] = record
            print(f"{problem.problem_id}: {'ok' if record.get('success') else record.get('error')}")
            persist()
        if not pending:
            persist()
        return 0

    worker_state = threading.local()

    def run_parallel(index: int, problem: K12Problem) -> tuple[int, Dict[str, Any]]:
        if not hasattr(worker_state, "pipeline"):
            worker_state.pipeline = pipeline_factory()
        return run_one(worker_state.pipeline, index, problem)

    print(f"parallel generation enabled: jobs={jobs}, pending={len(pending)}")
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="k12sim") as executor:
        futures = {
            executor.submit(run_parallel, index, problem): (index, problem)
            for index, problem in pending
        }
        for future in as_completed(futures):
            index, problem = futures[future]
            try:
                index, record = future.result()
            except Exception as exc:
                # Pipeline implementations normally return a failure manifest. This guard
                # keeps an unexpected worker failure from terminating the whole benchmark.
                record = {
                    "problem_id": problem.problem_id,
                    "model": args.model,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "diagnostics": {"worker_failure": True},
                }
                item_dir = output / safe_artifact_name(problem.problem_id)
                write_json(item_dir / "manifest.json", record)
            records[index] = record
            print(f"{problem.problem_id}: {'ok' if record.get('success') else record.get('error')}")
            persist()
    if not pending:
        persist()
    return 0


def command_validate(args: argparse.Namespace) -> int:
    failures = 0
    for record in read_records(args.benchmark):
        try:
            K12Problem.from_record(record)
        except Exception as exc:
            failures += 1
            print(f"invalid benchmark record: {exc}", file=sys.stderr)
    if args.artifacts:
        for run_dir in sorted(Path(args.artifacts).iterdir()):
            if not run_dir.is_dir() or not (run_dir / "program.json").exists():
                continue
            try:
                story = json.loads((run_dir / "storyboard.json").read_text(encoding="utf-8"))
                spec = EduWorldSpec.from_dict(json.loads((run_dir / "world_spec.json").read_text(encoding="utf-8")))
                from .models import StoryBlock

                blocks = [StoryBlock.from_dict(item) for item in story["blocks"]]
                program = json.loads((run_dir / "program.json").read_text(encoding="utf-8"))
                report = validate_program_payload(program, spec, blocks)
                if not report.valid:
                    failures += 1
                    print(f"{run_dir.name}: {'; '.join(report.errors)}", file=sys.stderr)
                manifest_file = run_dir / "manifest.json"
                manifest = (
                    json.loads(manifest_file.read_text(encoding="utf-8"))
                    if manifest_file.exists()
                    else {}
                )
                if manifest.get("method") == "k12simworld_candidate_target":
                    required = [
                        "solution_spec.json",
                        "simulation_contract.json",
                        "target_validation.json",
                    ]
                    missing = [name for name in required if not (run_dir / name).is_file()]
                    if missing:
                        raise ValueError(f"missing candidate-target artifacts: {missing}")
                    target_validation = json.loads(
                        (run_dir / "target_validation.json").read_text(encoding="utf-8")
                    )
                    if str(program.get("engine") or "") in DOMAIN_ENGINES:
                        if not (run_dir / "observed_trace.json").is_file():
                            raise ValueError("missing observed_trace.json for trusted domain engine")
                        if target_validation.get("status") != "passed":
                            raise ValueError(
                                "candidate target validation is not passed: "
                                f"{target_validation.get('status')}"
                            )
            except Exception as exc:
                failures += 1
                print(f"{run_dir.name}: {exc}", file=sys.stderr)
    print(f"validation failures: {failures}")
    return 1 if failures else 0


def command_evaluate(args: argparse.Namespace) -> int:
    records = [dict(row) for row in read_records(args.manifests)]
    external: Dict[str, Dict[str, Any]] = {}
    for score_path in args.scores or []:
        for row in read_records(score_path):
            problem_id = str(row.get("problem_id") or "")
            if not problem_id:
                continue
            bucket = external.setdefault(problem_id, {"scores": {}, "diagnostics": {}})
            bucket["scores"].update(dict(row.get("scores") or {}))
            if "trajectory_nrmse" in bucket["scores"]:
                bucket["diagnostics"]["expert_trajectory_nrmse"] = bucket["scores"].pop(
                    "trajectory_nrmse"
                )
    for record in records:
        supplement = external.get(str(record.get("problem_id") or ""))
        if not supplement:
            continue
        record["scores"] = {
            **dict(record.get("scores") or {}),
            **dict(supplement["scores"]),
        }
        record["diagnostics"] = {
            **dict(record.get("diagnostics") or {}),
            **dict(supplement["diagnostics"]),
        }
    outputs = write_evaluation_report(records, args.output_dir)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


def command_score_traces(args: argparse.Namespace) -> int:
    references = {str(row["problem_id"]): row for row in read_records(args.references)}
    missing_observed = []
    if args.observed:
        observations = [dict(row) for row in read_records(args.observed)]
    else:
        observations = []
        manifest_root = Path(args.manifests).resolve().parent
        for manifest in read_records(args.manifests):
            problem_id = str(manifest.get("problem_id") or "")
            raw_path = str(manifest.get("observed_trace_path") or "")
            if not raw_path:
                missing_observed.append(problem_id)
                continue
            trace_path = Path(raw_path)
            if not trace_path.is_absolute():
                trace_path = manifest_root / trace_path
            try:
                observed = json.loads(trace_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                missing_observed.append(problem_id)
                continue
            if isinstance(observed, dict):
                observations.append(observed)
            else:
                missing_observed.append(problem_id)

    output = []
    missing_references = []
    for observed in observations:
        problem_id = str(observed.get("problem_id") or "")
        reference = references.get(problem_id)
        if reference is None:
            missing_references.append(problem_id)
            continue
        output.append({"problem_id": problem_id, "scores": score_trace(reference, observed)})
    write_jsonl(args.output, output)
    summary = {
        "scored": len(output),
        "missing_references": missing_references,
        "missing_observed_traces": missing_observed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if args.strict and (missing_references or missing_observed) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="k12simworld")
    sub = parser.add_subparsers(dest="command", required=True)

    curate = sub.add_parser("curate", help="curate K12Vista-compatible JSON/JSONL")
    curate.add_argument("--input", required=True)
    curate.add_argument("--output-dir", required=True)
    curate.add_argument("--physics-target", type=int, default=180)
    curate.add_argument("--extension-target", type=int, default=60)
    curate.add_argument("--expert-target", type=int, default=60)
    curate.add_argument("--seed", type=int, default=2026)
    curate.set_defaults(func=command_curate)

    prepare = sub.add_parser("prepare-prompts", help="materialise gold-free storyboard prompts")
    prepare.add_argument("--benchmark", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--limit", type=int, default=0)
    prepare.set_defaults(func=command_prepare)

    human = sub.add_parser(
        "prepare-human-selection",
        help="map a human-curated K12Vista ID list to an experiment benchmark",
    )
    human.add_argument("--input", required=True, help="raw K12Vista JSON/JSONL")
    human.add_argument("--selection", required=True, help="versioned grouped ID JSON")
    human.add_argument("--screening-results", help="optional full screening_results.jsonl")
    human.add_argument("--output-dir", required=True)
    human.add_argument("--smoke-target", type=int, default=20)
    human.add_argument("--seed", type=int, default=2026)
    human.add_argument("--allow-integrity-errors", action="store_true")
    human.set_defaults(func=command_prepare_human_selection)

    tiers = sub.add_parser(
        "partition-physics-tiers",
        help="partition a human-selected physics benchmark by execution readiness",
    )
    tiers.add_argument("--benchmark", required=True)
    tiers.add_argument("--output-dir", required=True)
    tiers.set_defaults(func=command_partition_physics_tiers)

    domain = sub.add_parser(
        "simulate-domain",
        help="execute one declarative mechanics/equation/circuit/optics spec without an LLM",
    )
    domain.add_argument("--engine", required=True, choices=sorted(DOMAIN_ENGINES))
    domain.add_argument("--spec", required=True)
    domain.add_argument("--output-dir", required=True)
    domain.add_argument("--render", action="store_true")
    domain.set_defaults(func=command_simulate_domain)

    generate = sub.add_parser("generate", help="call a configured model and optionally render")
    generate.add_argument("--benchmark", required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument(
        "--engine",
        choices=(
            "threejs-cannon", "p5js", "manim", "mechanics-2d",
            "equation-solver", "circuit-solver", "ray-optics",
        ),
    )
    generate.add_argument("--limit", type=int, default=0)
    generate.add_argument("--render", action="store_true")
    generate.add_argument(
        "--jobs", type=int, default=1,
        help="parallel problems (start with 2 for remote APIs; default: 1)",
    )
    generate.add_argument("--resume", action="store_true", help="skip items with an existing manifest")
    generate.add_argument("--retry-failed", action="store_true", help="with --resume, rerun existing failures")
    generate.set_defaults(func=command_generate)

    baseline = sub.add_parser("generate-baseline", help="run one controlled experimental baseline")
    baseline.add_argument("--benchmark", required=True)
    baseline.add_argument("--output-dir", required=True)
    baseline.add_argument("--model", required=True)
    baseline.add_argument("--method", required=True, choices=sorted(BASELINE_METHODS))
    baseline.add_argument("--engine", choices=("threejs-cannon", "p5js", "manim"))
    baseline.add_argument("--limit", type=int, default=0)
    baseline.add_argument("--render", action="store_true")
    baseline.add_argument(
        "--jobs", type=int, default=1,
        help="parallel problems (default: 1)",
    )
    baseline.add_argument("--resume", action="store_true", help="skip items with an existing manifest")
    baseline.add_argument("--retry-failed", action="store_true", help="with --resume, rerun existing failures")
    baseline.set_defaults(func=command_generate_baseline)

    validate = sub.add_parser("validate", help="validate benchmark and generated contracts")
    validate.add_argument("--benchmark", required=True)
    validate.add_argument("--artifacts")
    validate.set_defaults(func=command_validate)

    evaluate = sub.add_parser("evaluate", help="score manifests and create paper tables")
    evaluate.add_argument("--manifests", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument(
        "--scores",
        action="append",
        help="optional JSON/JSONL scores to merge by problem_id; may be repeated",
    )
    evaluate.set_defaults(func=command_evaluate)

    traces = sub.add_parser("score-traces", help="compare observed state traces with expert references")
    traces.add_argument("--references", required=True)
    observed_source = traces.add_mutually_exclusive_group(required=True)
    observed_source.add_argument("--observed", help="normalized observed trace JSON/JSONL")
    observed_source.add_argument(
        "--manifests", help="generation manifests containing observed_trace_path"
    )
    traces.add_argument("--output", required=True)
    traces.add_argument("--strict", action="store_true")
    traces.set_defaults(func=command_score_traces)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
