"""Turn a human-curated ID list into an experiment-ready K12SimBench manifest."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .curation import assign_knowledge_disjoint_splits
from .io import read_records, write_json, write_jsonl
from .models import K12Problem


GROUP_FALLBACK_FAMILY = {
    "运动相关": "kinematics",
    "力学": "force_and_motion",
    "碰撞": "collision",
    "简谐振动": "spring",
    "能量": "energy",
    "电磁场": "charged_particle",
    "电路": "circuit",
    "光学": "ray_optics",
    "压强": "buoyancy",
}

# These families are supported by the first-version deterministic Canvas or
# Three.js/Cannon renderers without requiring a separate domain solver.
SMOKE_FAMILIES = {
    "kinematics", "projectile", "force_and_motion", "inclined_plane",
    "friction", "lever", "pulley", "spring", "collision",
    "circular_motion", "energy", "buoyancy",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _load_selection(path: Path) -> tuple[List[str], Dict[str, List[str]], Dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    groups = value.get("groups")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("selection config requires a non-empty groups object")
    ordered: List[str] = []
    memberships: Dict[str, List[str]] = defaultdict(list)
    for group, raw_ids in groups.items():
        if not isinstance(raw_ids, list):
            raise ValueError(f"selection group {group!r} must be a list")
        for raw_id in raw_ids:
            item_id = str(raw_id).strip().lower()
            if len(item_id) != 64 or any(char not in "0123456789abcdef" for char in item_id):
                raise ValueError(f"invalid K12Vista hash_id in {group!r}: {raw_id!r}")
            if str(group) not in memberships[item_id]:
                memberships[item_id].append(str(group))
            if item_id not in ordered:
                ordered.append(item_id)
    return ordered, dict(memberships), value


def _screening_subset(path: Path | None, wanted: set[str]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for row in read_records(path):
        item_id = str(row.get("id") or row.get("hash_id") or "").lower()
        if item_id in wanted:
            output[item_id] = row
    return output


def _selected_raw_records(path: Path, wanted: set[str]):
    """Decode only selected K12Vista rows.

    K12Vista stores ``hash_id`` first and then a large Base64 image. Extracting
    the leading ID before ``json.loads`` avoids decoding every unselected image.
    The fallback preserves compatibility with JSONL files that use another key
    order.
    """
    id_pattern = re.compile(r'"(?:hash_id|problem_id|id)"\s*:\s*"([^"]+)"')
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            match = id_pattern.search(line[:512])
            if match:
                if match.group(1).strip().lower() not in wanted:
                    continue
                row = json.loads(line)
            else:
                row = json.loads(line)
                item_id = str(
                    row.get("hash_id") or row.get("problem_id") or row.get("id") or ""
                ).strip().lower()
                if item_id not in wanted:
                    continue
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def _family(screening: Mapping[str, Any], groups: Sequence[str]) -> str:
    value = str(screening.get("simulation_family") or "").strip()
    if value and value not in {"none", "static_diagram", "other"}:
        return value
    for group in groups:
        if group in GROUP_FALLBACK_FAMILY:
            return GROUP_FALLBACK_FAMILY[group]
    return "other"


def _smoke_subset(records: Sequence[Dict[str, Any]], target: int, seed: int) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = record.get("source_metadata") or {}
        screening = metadata.get("screening") or {}
        family = str(record.get("simulation_type") or "")
        if family not in SMOKE_FAMILIES or not record.get("image"):
            continue
        if screening.get("manual_review_required") or screening.get("conflict_flag"):
            continue
        if screening.get("final_category") not in {"A_CORE", "B_EXTENSION"}:
            continue
        group = str((metadata.get("human_selection_groups") or ["未分组"])[0])
        buckets[group].append(record)
    for group in buckets:
        buckets[group].sort(
            key=lambda row: (
                0 if (row.get("source_metadata") or {}).get("screening", {}).get("final_category") == "A_CORE" else 1,
                -float((row.get("source_metadata") or {}).get("screening", {}).get("confidence") or 0),
                _stable(seed, str(row["problem_id"])),
            )
        )
    chosen: List[Dict[str, Any]] = []
    active = [group for group in GROUP_FALLBACK_FAMILY if buckets.get(group)]
    while active and len(chosen) < target:
        remaining: List[str] = []
        for group in active:
            if buckets[group] and len(chosen) < target:
                chosen.append(buckets[group].pop(0))
            if buckets[group]:
                remaining.append(group)
        active = remaining
    return chosen


def prepare_human_physics_selection(
    raw_path: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    *,
    screening_path: str | Path | None = None,
    smoke_target: int = 20,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Map human-selected IDs to K12Vista and materialise reproducible manifests."""
    raw = Path(raw_path)
    selection = Path(selection_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ordered_ids, memberships, selection_config = _load_selection(selection)
    wanted = set(ordered_ids)
    screening = _screening_subset(Path(screening_path) if screening_path else None, wanted)

    raw_matches: Dict[str, Dict[str, Any]] = {}
    raw_duplicate_ids: List[str] = []
    for row in _selected_raw_records(raw, wanted):
        item_id = str(row.get("hash_id") or row.get("problem_id") or row.get("id") or "").strip().lower()
        if item_id not in wanted:
            continue
        if item_id in raw_matches:
            raw_duplicate_ids.append(item_id)
            continue
        raw_matches[item_id] = row

    missing_ids = [item_id for item_id in ordered_ids if item_id not in raw_matches]
    nonphysics_ids: List[str] = []
    records: List[Dict[str, Any]] = []
    for order, item_id in enumerate(ordered_ids, 1):
        row = raw_matches.get(item_id)
        if row is None:
            continue
        problem = K12Problem.from_record(row)
        if problem.subject != "physics":
            nonphysics_ids.append(item_id)
            continue
        screen = screening.get(item_id, {})
        override = dict((selection_config.get("overrides") or {}).get(item_id) or {})
        family = str(override.get("simulation_type") or _family(screen, memberships[item_id]))
        record = problem.benchmark_record()
        record["simulation_type"] = family
        record["dynamic_suitability"] = True
        record["source_metadata"] = {
            **dict(record.get("source_metadata") or {}),
            "dataset": "K12Vista",
            "raw_subject": row.get("subject"),
            "human_selected": True,
            "human_selection_order": order,
            "human_selection_groups": memberships[item_id],
            "selection_source": selection_config.get("source") or {},
            **({"human_override": override} if override else {}),
            "screening": {
                key: screen.get(key)
                for key in (
                    "final_category", "decision_source", "simulation_family",
                    "recommended_backend", "scores", "manual_review_required",
                    "conflict_flag", "confidence", "selection_reason",
                )
                if key in screen
            },
        }
        records.append(record)

    problems = [K12Problem.from_record(record) for record in records]
    splits = assign_knowledge_disjoint_splits(problems, seed=seed)
    for record in records:
        record["split"] = splits[str(record["problem_id"])]
        # Expert executable references are a separate annotation step. Human
        # selection alone must not pretend that a gold state trace exists.
        record["expert_reference"] = False

    smoke = _smoke_subset(records, max(0, smoke_target), seed)
    smoke_ids = [str(record["problem_id"]) for record in smoke]
    benchmark_path = output / "physics_k12simbench.jsonl"
    smoke_path = output / f"physics_smoke_{len(smoke)}.jsonl"
    write_jsonl(benchmark_path, records)
    write_jsonl(smoke_path, smoke)
    write_jsonl(
        output / "physics_model_input_preview.jsonl",
        ({**K12Problem.from_record(record).model_payload(), "image_present": bool(record.get("image"))} for record in records),
    )
    write_json(output / "physics_smoke_ids.json", smoke_ids)

    duplicate_memberships = {
        item_id: groups for item_id, groups in memberships.items() if len(groups) > 1
    }
    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "raw_path": str(raw.resolve()),
        "selection_path": str(selection.resolve()),
        "screening_path": str(Path(screening_path).resolve()) if screening_path else None,
        "selection_occurrences": sum(len(ids) for ids in selection_config["groups"].values()),
        "selection_unique_ids": len(ordered_ids),
        "matched_physics": len(records),
        "missing_ids": missing_ids,
        "nonphysics_ids": nonphysics_ids,
        "raw_duplicate_ids": sorted(set(raw_duplicate_ids)),
        "duplicate_group_memberships": duplicate_memberships,
        "screening_records_matched": sum(1 for item_id in ordered_ids if item_id in screening),
        "manual_overrides": selection_config.get("overrides") or {},
        "smoke_items": len(smoke),
        "smoke_ids": smoke_ids,
        "by_human_group": dict(sorted(Counter(group for record in records for group in record["source_metadata"]["human_selection_groups"]).items())),
        "by_grade": dict(sorted(Counter(str(record.get("grade") or "unknown") for record in records).items())),
        "by_question_type": dict(sorted(Counter(str(record.get("question_type") or "unknown") for record in records).items())),
        "by_simulation_type": dict(sorted(Counter(str(record.get("simulation_type") or "other") for record in records).items())),
        "by_screening_category": dict(sorted(Counter(str((record["source_metadata"].get("screening") or {}).get("final_category") or "unavailable") for record in records).items())),
        "by_split": dict(sorted(Counter(str(record["split"]) for record in records).items())),
    }
    write_json(output / "physics_selection_report.json", report)
    checksums = {
        benchmark_path.name: _sha256(benchmark_path),
        smoke_path.name: _sha256(smoke_path),
        selection.name: _sha256(selection),
    }
    write_json(output / "checksums.json", checksums)
    (output / "README.md").write_text(
        "# K12SimWorld physics benchmark\n\n"
        "`physics_k12simbench.jsonl` is the complete deduplicated human-selected physics set. "
        f"`{smoke_path.name}` is a deterministic first-run subset restricted to renderer-supported "
        "families and high-confidence multimodal screening results. Gold answers remain in the "
        "benchmark for evaluation, but `K12Problem.model_payload()` excludes them from model calls. "
        "`physics_model_input_preview.jsonl` can be audited to verify this separation.\n",
        encoding="utf-8",
    )
    return report
