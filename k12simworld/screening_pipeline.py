"""Full-dataset K12Vista screening with deterministic audit and two model stages.

The source dataset is opened read-only. SQLite is the authoritative checkpoint;
published JSONL/CSV/HTML files are deterministic exports from that checkpoint.
Secrets are read from the environment and are never persisted or logged.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import csv
import hashlib
import html
import io
import json
import math
import os
import random
import re
import shutil
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from PIL import Image, ImageStat, UnidentifiedImageError


SCREENING_VERSION = "k12vista-executable-screen-v1.0"
TEXT_PROMPT_VERSION = "deepseek-prefilter-v1.0"
VISION_PROMPT_VERSION = "qwen-multimodal-final-v1.0"
DEFAULT_SEED = 2026
SOURCE_RELATIVE = "raw/K12_Vista.jsonl"

PREFILTER_LABELS = {"likely_dynamic", "uncertain", "likely_static", "obvious_reject"}
FINAL_CATEGORIES = {"A_CORE", "B_EXTENSION", "C_STATIC", "D_REJECT"}
SIMULATION_FAMILIES = {
    "kinematics", "projectile", "force_and_motion", "inclined_plane", "friction",
    "lever", "pulley", "spring", "collision", "circular_motion", "energy",
    "charged_particle", "circuit", "ray_optics", "buoyancy", "chemistry_process",
    "biology_process", "geography_process", "static_diagram", "other", "none",
}
BACKENDS = {
    "canvas_2d", "svg", "matterjs", "p5js", "threejs_cannon", "circuit_solver",
    "ray_optics", "manim", "custom_rule_engine", "other", "none",
}
SCORE_NAMES = (
    "dynamic_process", "explicit_rule", "initial_condition", "outcome_verifiable",
    "engine_feasibility", "visual_dependency", "educational_value", "data_quality",
)

TEXT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id", "prefilter_label", "confidence", "image_needed", "dynamic_evidence",
        "rule_evidence", "static_evidence", "possible_simulation_family",
        "missing_information", "reason",
    ],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "prefilter_label": {"enum": sorted(PREFILTER_LABELS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "image_needed": {"type": "boolean"},
        "dynamic_evidence": {"type": "array", "items": {"type": "string"}},
        "rule_evidence": {"type": "array", "items": {"type": "string"}},
        "static_evidence": {"type": "array", "items": {"type": "string"}},
        "possible_simulation_family": {"type": "array", "items": {"type": "string"}},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string", "minLength": 1},
    },
}

VISION_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id", "final_category", "simulation_family", "recommended_backend", "scores",
        "required_objects", "initial_state", "governing_rules", "dynamic_description",
        "expected_observable", "answer_verification", "missing_information",
        "reasonable_defaults", "conflict_flag", "conflict_description", "selection_reason",
        "manual_review_required", "confidence",
    ],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "final_category": {"enum": sorted(FINAL_CATEGORIES)},
        "simulation_family": {"enum": sorted(SIMULATION_FAMILIES)},
        "recommended_backend": {"enum": sorted(BACKENDS)},
        "scores": {
            "type": "object", "additionalProperties": False,
            "required": list(SCORE_NAMES),
            "properties": {name: {"type": "integer", "minimum": 0, "maximum": 3} for name in SCORE_NAMES},
        },
        "required_objects": {"type": "array", "items": {"type": "string"}},
        "initial_state": {"type": "array", "items": {"type": "string"}},
        "governing_rules": {"type": "array", "items": {"type": "string"}},
        "dynamic_description": {"type": "string"},
        "expected_observable": {"type": "string"},
        "answer_verification": {"type": "string"},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "reasonable_defaults": {"type": "array", "items": {"type": "string"}},
        "conflict_flag": {"type": "boolean"},
        "conflict_description": {"type": "string"},
        "selection_reason": {"type": "string", "minLength": 1},
        "manual_review_required": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

TEXT_VALIDATOR = Draft202012Validator(TEXT_SCHEMA)
VISION_VALIDATOR = Draft202012Validator(VISION_SCHEMA)

TEXT_SYSTEM_PROMPT = """你是一名K12学科教育和可执行仿真数据筛选专家。
你的任务是根据题目文本、学科、知识点、标准答案和解析，对样本进行高召回率预筛选。
你不是最终评审模型，不得输出A_CORE或B_EXTENSION；你只判断该题是否值得进一步查看原始题图。
重点判断对象运动、系统状态变化、对象交互、可编码学科规律、实验操作或参数过程、
仿真能否帮助验证答案，以及是否必须查看真实题图。严格区分真实状态演化、静态关系、
只有标注价值的内容和纯记忆/定义/简单代入题。采取高召回策略，不确定时输出uncertain。
img_caption只是辅助文本，不能视为你看到了真实题图。只输出符合指定Schema的JSON。"""

VISION_SYSTEM_PROMPT = """你是一名K12学科教育、多模态理解和可执行科学仿真专家。
你的任务不是解题，而是判断题目是否适合评测“从题目和题图生成可执行仿真”的能力。
结合实际题图、题目、学科、知识点、标准答案和解析，识别对象、空间/逻辑关系、初始状态、
状态变化、可编码规律、可执行性、答案可验证性、动态教育增益和工具实现稳定性。
严格区分真实动态过程/状态转换与静态标注、箭头、高亮、辅助线、逐步显示和装饰动画。
图与caption冲突时以实际图片为主要证据并记录冲突。不得虚构关键参数；不确定时要求人工复核。
只输出符合指定Schema的JSON，不输出Markdown、代码围栏或额外解释。"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_dumps(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2 if pretty else None, separators=None if pretty else (",", ":"))


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[;；]", value) if part.strip()]
    if isinstance(value, Mapping):
        return [json_dumps(value)]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def first_value(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def truncate(value: Any, limit: int = 16000) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json_dumps(value)
    return text if len(text) <= limit else text[:limit] + "…[truncated]"


def parse_grade(subject: str, explicit: Any = None) -> str:
    if explicit not in (None, ""):
        return str(explicit)
    match = re.search(r"(?:^|[-_])g(?:rade)?\s*(\d{1,2})\b", str(subject), re.I)
    return match.group(1) if match else "unknown"


def base_subject(subject: str) -> str:
    value = str(subject or "unknown").strip()
    return re.split(r"[-_]g(?:rade)?\s*\d{1,2}\b", value, flags=re.I)[0].lower() or "unknown"


@dataclass(frozen=True)
class Config:
    source: Path
    output: Path
    deepseek_key: str
    deepseek_base: str
    deepseek_model: str
    qwen_key: str
    qwen_base: str
    qwen_model: str
    seed: int
    deepseek_workers: int
    qwen_workers: int
    max_retries: int

    @classmethod
    def load(cls, args: argparse.Namespace) -> "Config":
        load_dotenv(Path(args.project_root) / ".env")
        deepseek_key = (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        deepseek_base = (os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com").strip()
        qwen_key = (os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or "").strip()
        qwen_base = (os.getenv("DASHSCOPE_BASE_URL") or os.getenv("DASHSCOPE_API_BASE") or os.getenv("QWEN_API_BASE") or "").strip()
        deepseek_model = (os.getenv("DEEPSEEK_TEXT_MODEL") or "").strip()
        qwen_model = (os.getenv("QWEN_VL_MODEL") or "").strip()
        missing = [name for name, value in (
            ("DEEPSEEK_API_KEY/OPENAI_API_KEY", deepseek_key),
            ("DEEPSEEK_TEXT_MODEL", deepseek_model),
            ("DASHSCOPE_API_KEY", qwen_key),
            ("DASHSCOPE_BASE_URL/DASHSCOPE_API_BASE", qwen_base),
            ("QWEN_VL_MODEL", qwen_model),
        ) if not value]
        if missing:
            raise RuntimeError("missing required environment configuration: " + ", ".join(missing))
        source = Path(args.input).resolve()
        if not source.is_file():
            raise RuntimeError(f"dataset is not a readable file: {source}")
        return cls(
            source=source, output=Path(args.output).resolve(), deepseek_key=deepseek_key,
            deepseek_base=deepseek_base, deepseek_model=deepseek_model,
            qwen_key=qwen_key, qwen_base=qwen_base, qwen_model=qwen_model,
            seed=args.seed, deepseek_workers=max(1, args.deepseek_workers),
            qwen_workers=max(1, args.qwen_workers), max_retries=max(1, args.max_retries),
        )


class Checkpoint:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.local = threading.local()
        conn = sqlite3.connect(path)
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS quality (
                id TEXT PRIMARY KEY, line_number INTEGER, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS stage (
                stage TEXT NOT NULL, id TEXT NOT NULL, status TEXT NOT NULL,
                payload TEXT, raw_response TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
                latency REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL,
                PRIMARY KEY(stage,id)
            );
            CREATE TABLE IF NOT EXISTS errors (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT, stage TEXT NOT NULL,
                error_type TEXT NOT NULL, message TEXT NOT NULL, attempt INTEGER,
                timestamp TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        conn.commit(); conn.close()

    def connection(self) -> sqlite3.Connection:
        conn = getattr(self.local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60)
            conn.execute("PRAGMA busy_timeout=60000")
            self.local.conn = conn
        return conn

    def put_quality(self, item_id: str, line_number: int, payload: Mapping[str, Any]) -> None:
        conn = self.connection()
        conn.execute("INSERT OR REPLACE INTO quality(id,line_number,payload) VALUES(?,?,?)", (item_id, line_number, json_dumps(payload)))
        conn.commit()

    def quality_map(self) -> Dict[str, Dict[str, Any]]:
        return {row[0]: json.loads(row[1]) for row in self.connection().execute("SELECT id,payload FROM quality")}

    def completed_ids(self, stage: str) -> set[str]:
        return {row[0] for row in self.connection().execute("SELECT id FROM stage WHERE stage=? AND status='success'", (stage,))}

    def stage_map(self, stage: str) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for item_id, status, payload, attempts, inp, out, latency in self.connection().execute(
            "SELECT id,status,payload,attempts,input_tokens,output_tokens,latency FROM stage WHERE stage=?", (stage,)
        ):
            result[item_id] = {
                "status": status, "payload": json.loads(payload) if payload else None,
                "attempts": attempts, "input_tokens": inp, "output_tokens": out, "latency": latency,
            }
        return result

    def put_stage(self, stage: str, item_id: str, status: str, payload: Optional[Mapping[str, Any]], raw: str,
                  attempts: int, input_tokens: int, output_tokens: int, latency: float) -> None:
        conn = self.connection()
        conn.execute(
            "INSERT OR REPLACE INTO stage(stage,id,status,payload,raw_response,attempts,input_tokens,output_tokens,latency,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (stage, item_id, status, json_dumps(payload) if payload is not None else None, raw, attempts, input_tokens, output_tokens, latency, utc_now()),
        )
        conn.commit()

    def error(self, item_id: Optional[str], stage: str, error_type: str, message: str, attempt: int = 0) -> None:
        conn = self.connection()
        conn.execute("INSERT INTO errors(id,stage,error_type,message,attempt,timestamp) VALUES(?,?,?,?,?,?)",
                     (item_id, stage, error_type, message[:8000], attempt, utc_now()))
        conn.commit()

    def set_meta(self, key: str, value: Any) -> None:
        conn = self.connection()
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, json_dumps(value)))
        conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.connection().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default


def iter_source(path: Path) -> Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                yield line_number, None, "blank line"
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not an object")
                yield line_number, value, None
            except Exception as exc:
                yield line_number, None, f"{type(exc).__name__}: {exc}"


FIELD_CANDIDATES = {
    "id": ["hash_id", "problem_id", "id", "index"],
    "question": ["question", "problem", "prompt", "text"],
    "image": ["img", "image", "image_path", "figure"],
    "image_caption": ["img_caption", "image_caption", "caption"],
    "subject": ["subject", "discipline"],
    "grade": ["grade", "level"],
    "question_type": ["type", "question_type"],
    "knowledge_points": ["knowledge_point", "knowledge_points", "concepts"],
    "answer_container": ["format_answer", "answer"],
    "ground_truth": ["ground_truth", "final_answer"],
    "solution": ["format_solution", "solution", "reference_solution", "analysis"],
    "difficulty": ["difficulty"],
}


def normalize_record(record: Mapping[str, Any], line_number: int) -> Dict[str, Any]:
    answer = first_value(record, FIELD_CANDIDATES["answer_container"])
    answer_map = answer if isinstance(answer, Mapping) else {}
    raw_id = first_value(record, FIELD_CANDIDATES["id"])
    question = str(first_value(record, FIELD_CANDIDATES["question"]) or "").strip()
    item_id = str(raw_id or f"missing-id-line-{line_number}").strip()
    subject = str(first_value(record, FIELD_CANDIDATES["subject"]) or "unknown").strip()
    ground = first_value(record, FIELD_CANDIDATES["ground_truth"])
    if ground is None:
        ground = first_value(answer_map, FIELD_CANDIDATES["ground_truth"])
    solution = first_value(record, FIELD_CANDIDATES["solution"])
    if solution is None:
        solution = first_value(answer_map, FIELD_CANDIDATES["solution"])
    return {
        "id": item_id, "raw_id_present": raw_id not in (None, ""), "line_number": line_number,
        "question": question, "image": first_value(record, FIELD_CANDIDATES["image"]),
        "image_caption": str(first_value(record, FIELD_CANDIDATES["image_caption"]) or "").strip(),
        "subject": subject, "subject_base": base_subject(subject),
        "grade": parse_grade(subject, first_value(record, FIELD_CANDIDATES["grade"])),
        "question_type": str(first_value(record, FIELD_CANDIDATES["question_type"]) or "unknown").strip(),
        "knowledge_points": as_list(first_value(record, FIELD_CANDIDATES["knowledge_points"])),
        "ground_truth": as_list(ground), "solution": as_list(solution),
        "difficulty": str(first_value(record, FIELD_CANDIDATES["difficulty"]) or "unknown"),
    }


def decode_image(raw: Any) -> Tuple[Optional[bytes], str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"present": bool(raw), "decodable": False, "blank_flag": False}
    if not raw:
        return None, "", meta
    try:
        if isinstance(raw, str) and raw.startswith("data:image/") and ";base64," in raw:
            header, encoded = raw.split(",", 1)
            media = header[5:].split(";", 1)[0]
            data = base64.b64decode(encoded, validate=False)
        elif isinstance(raw, str) and len(raw) < 4096 and Path(raw).is_file():
            path = Path(raw); data = path.read_bytes(); media = Image.MIME.get(Image.open(path).format, "image/png")
        elif isinstance(raw, str):
            data = base64.b64decode(raw, validate=False); media = ""
        else:
            raise ValueError("unsupported image value")
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            media = media or Image.MIME.get(image.format, "image/png")
            gray = image.convert("L")
            stat = ImageStat.Stat(gray)
            extrema = gray.getextrema()
            variance = float(stat.var[0]) if stat.var else 0.0
            meta.update({
                "decodable": True, "format": image.format, "width": image.width, "height": image.height,
                "bytes": len(data), "variance": round(variance, 4),
                "blank_flag": bool(variance < 0.5 or (extrema and extrema[1] - extrema[0] < 3)),
            })
        return data, media, meta
    except (ValueError, OSError, UnidentifiedImageError, base64.binascii.Error) as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return None, "", meta


def question_depends_on_image(question: str) -> bool:
    text = question.lower()
    return any(token in text for token in ("如图", "下图", "图中", "根据图", "shown", "figure", "diagram", "graph"))


def corrupted_text(question: str) -> bool:
    if not question.strip():
        return True
    replacement_ratio = question.count("�") / max(1, len(question))
    control = sum(ord(ch) < 32 and ch not in "\n\r\t" for ch in question)
    return replacement_ratio > 0.05 or control > max(3, len(question) // 20)


def audit_dataset(cfg: Config, db: Checkpoint) -> Dict[str, Any]:
    if db.get_meta("audit_complete", False):
        return db.get_meta("dataset_summary", {})
    cfg.output.mkdir(parents=True, exist_ok=True)
    counts = Counter(); subjects = Counter(); grades = Counter(); qtypes = Counter()
    missing = Counter(); seen_ids: Dict[str, int] = {}; seen_exact: Dict[str, str] = {}
    observed_keys = Counter(); invalid_lines = 0
    reservoir: List[str] = []; rng = random.Random(cfg.seed)
    for line_number, raw, parse_error in iter_source(cfg.source):
        counts["raw_records"] += 1
        if raw is None:
            invalid_lines += 1
            db.error(None, "data_quality", "json_error", f"line {line_number}: {parse_error}")
            continue
        observed_keys.update(raw.keys())
        item = normalize_record(raw, line_number)
        item_id = item["id"]
        if item_id in seen_ids:
            item_id = f"{item_id}__duplicate_id_line_{line_number}"
            item["id"] = item_id
            item["duplicate_id_of"] = str(first_value(raw, FIELD_CANDIDATES["id"]) or "")
        seen_ids[item_id] = line_number
        data, media, image_meta = decode_image(item["image"])
        exact_key = stable_hash("\u241f".join((item["question"], str(item["subject"]), hashlib.sha256(data or b"").hexdigest())))
        duplicate_of = seen_exact.get(exact_key)
        if duplicate_of is None:
            seen_exact[exact_key] = item_id
        hard_reasons: List[str] = []
        if not item["raw_id_present"]: hard_reasons.append("missing original id")
        if corrupted_text(item["question"]): hard_reasons.append("empty or severely corrupted question")
        if duplicate_of: hard_reasons.append("exact duplicate")
        if image_meta.get("present") and not image_meta.get("decodable") and question_depends_on_image(item["question"]):
            hard_reasons.append("required image is damaged")
        if not item["question"]: missing["question"] += 1
        if not image_meta.get("present"): missing["image"] += 1
        elif not image_meta.get("decodable"): missing["damaged_image"] += 1
        if image_meta.get("blank_flag"): missing["blank_image_flag"] += 1
        if not item["ground_truth"]: missing["answer"] += 1
        if not item["solution"]: missing["solution"] += 1
        if not item["knowledge_points"]: missing["knowledge_points"] += 1
        quality_score = 3
        if not image_meta.get("present") or not item["ground_truth"] or not item["solution"]: quality_score = min(quality_score, 2)
        if image_meta.get("blank_flag") or (image_meta.get("present") and not image_meta.get("decodable")): quality_score = min(quality_score, 1)
        if hard_reasons: quality_score = 0
        payload = {
            "id": item_id, "source_file": SOURCE_RELATIVE, "line_number": line_number,
            "subject": item["subject"], "subject_base": item["subject_base"], "grade": item["grade"],
            "question_type": item["question_type"], "knowledge_points": item["knowledge_points"],
            "question_present": bool(item["question"]), "answer_present": bool(item["ground_truth"]),
            "solution_present": bool(item["solution"]), "image": image_meta,
            "image_required_by_text": question_depends_on_image(item["question"]),
            "duplicate_of": duplicate_of, "hard_reject": bool(hard_reasons),
            "hard_reject_reasons": hard_reasons, "data_quality_score": quality_score,
        }
        db.put_quality(item_id, line_number, payload)
        counts["valid_json_records"] += 1
        counts["hard_reject"] += bool(hard_reasons)
        counts["exact_duplicates"] += bool(duplicate_of)
        subjects[item["subject"]] += 1; grades[item["grade"]] += 1; qtypes[item["question_type"]] += 1
        if len(reservoir) < 20: reservoir.append(item_id)
        else:
            j = rng.randrange(counts["valid_json_records"])
            if j < 20: reservoir[j] = item_id
        if counts["raw_records"] % 1000 == 0:
            print(f"[audit] {counts['raw_records']} records", flush=True)
    summary = {
        "screening_version": SCREENING_VERSION, "source_file": str(cfg.source), "source_bytes": cfg.source.stat().st_size,
        "raw_records": counts["raw_records"], "valid_json_records": counts["valid_json_records"],
        "invalid_json_records": invalid_lines, "hard_reject_records": counts["hard_reject"],
        "exact_duplicate_records": counts["exact_duplicates"], "by_subject": dict(sorted(subjects.items())),
        "by_grade": dict(sorted(grades.items())), "by_question_type": dict(sorted(qtypes.items())),
        "missing": dict(sorted(missing.items())), "random_audit_ids": reservoir, "seed": cfg.seed,
        "completed_at": utc_now(),
    }
    mapping = {
        "source_format": "jsonl", "image_storage": "base64 embedded in img (data URL also accepted)",
        "observed_fields": dict(sorted(observed_keys.items())), "canonical_mapping": FIELD_CANDIDATES,
        "answer_nested_mapping": {"container": "format_answer", "answer": "ground_truth", "solution": "format_solution"},
        "normalization_notes": ["grade inferred from subject suffix such as physics-g9", "original subject retained in outputs"],
    }
    (cfg.output / "dataset_summary.json").write_text(json_dumps(summary, pretty=True) + "\n", encoding="utf-8")
    (cfg.output / "field_mapping.json").write_text(json_dumps(mapping, pretty=True) + "\n", encoding="utf-8")
    (cfg.output / "audit_random20.json").write_text(json_dumps({"seed": cfg.seed, "ids": reservoir}, pretty=True) + "\n", encoding="utf-8")
    (cfg.output / "schemas").mkdir(exist_ok=True)
    (cfg.output / "schemas" / "deepseek_prefilter.schema.json").write_text(json_dumps(TEXT_SCHEMA, pretty=True) + "\n", encoding="utf-8")
    (cfg.output / "schemas" / "qwen_multimodal.schema.json").write_text(json_dumps(VISION_SCHEMA, pretty=True) + "\n", encoding="utf-8")
    db.set_meta("dataset_summary", summary); db.set_meta("field_mapping", mapping); db.set_meta("audit_complete", True)
    return summary


def parse_json_response(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response contains no JSON object")
        value = json.loads(raw[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value


def endpoint(base: str) -> str:
    value = base.rstrip("/")
    return value if value.endswith("/chat/completions") else value + "/chat/completions"


def usage_from(data: Mapping[str, Any]) -> Tuple[int, int]:
    usage = data.get("usage") or {}
    return int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0), int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)


_HTTP_LOCAL = threading.local()


def _http_session() -> requests.Session:
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _HTTP_LOCAL.session = session
    return session


def post_json(url: str, key: str, payload: Mapping[str, Any], timeout: int = 180) -> Tuple[Dict[str, Any], float]:
    started = time.monotonic()
    response = _http_session().post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=timeout)
    latency = time.monotonic() - started
    if response.status_code != 200:
        excerpt = (response.text or "")[:2000]
        raise RuntimeError(f"HTTP {response.status_code}: {excerpt}")
    return response.json(), latency


def text_prompt(item: Mapping[str, Any]) -> str:
    contract = {
        "id": item["id"], "prefilter_label": "likely_dynamic | uncertain | likely_static | obvious_reject",
        "confidence": 0.0, "image_needed": True, "dynamic_evidence": [], "rule_evidence": [],
        "static_evidence": [], "possible_simulation_family": [], "missing_information": [], "reason": "",
    }
    content = {
        "id": item["id"], "question": truncate(item["question"]), "subject": item["subject"],
        "grade": item["grade"], "question_type": item["question_type"], "knowledge_points": item["knowledge_points"],
        "standard_answer": item["ground_truth"], "solution": item["solution"],
        "img_caption_auxiliary_only": truncate(item["image_caption"], 8000),
    }
    return "请对以下样本进行高召回文本预筛。严格返回这个结构，不要增加字段：\n" + json_dumps(contract, pretty=True) + "\n样本：\n" + json_dumps(content, pretty=True)


def vision_prompt(item: Mapping[str, Any]) -> str:
    contract = {
        "id": item["id"], "final_category": "A_CORE | B_EXTENSION | C_STATIC | D_REJECT",
        "simulation_family": "one allowed family", "recommended_backend": "one allowed backend",
        "scores": {name: 0 for name in SCORE_NAMES}, "required_objects": [], "initial_state": [],
        "governing_rules": [], "dynamic_description": "", "expected_observable": "",
        "answer_verification": "", "missing_information": [], "reasonable_defaults": [],
        "conflict_flag": False, "conflict_description": "", "selection_reason": "",
        "manual_review_required": False, "confidence": 0.0,
    }
    content = {
        "id": item["id"], "question": truncate(item["question"]), "subject": item["subject"],
        "grade": item["grade"], "question_type": item["question_type"], "knowledge_points": item["knowledge_points"],
        "standard_answer": item["ground_truth"], "solution": item["solution"],
        "img_caption_auxiliary_only": truncate(item["image_caption"], 8000),
        "allowed_simulation_families": sorted(SIMULATION_FAMILIES), "allowed_backends": sorted(BACKENDS),
        "score_scale": "integer 0..3",
        "A_CORE_minimum": ["dynamic_process>=2", "explicit_rule>=2", "initial_condition>=2", "outcome_verifiable>=2", "engine_feasibility>=2", "educational_value>=2", "data_quality>=2"],
    }
    return "结合本消息中的真实图片进行最终判断。严格返回这个结构，不要增加字段：\n" + json_dumps(contract, pretty=True) + "\n样本：\n" + json_dumps(content, pretty=True)


def repair_instruction(raw: str, errors: Sequence[str], schema_name: str) -> str:
    return f"上一次{schema_name}输出无效。错误：{json_dumps(list(errors))}\n请只返回完整修复JSON。上次输出：\n{raw[:30000]}"


def validate_payload(value: Dict[str, Any], validator: Draft202012Validator, expected_id: str) -> None:
    errors = sorted(validator.iter_errors(value), key=lambda e: list(e.path))
    messages = [f"{'/'.join(map(str, err.path)) or '$'}: {err.message}" for err in errors]
    if str(value.get("id")) != expected_id: messages.append("id does not match source")
    if messages: raise ValueError("; ".join(messages[:20]))


def call_deepseek(cfg: Config, item: Mapping[str, Any]) -> Dict[str, Any]:
    messages = [{"role": "system", "content": TEXT_SYSTEM_PROMPT}, {"role": "user", "content": text_prompt(item)}]
    attempts = 0; retries = 0; total_latency = 0.0; last_raw = ""; inp = out = 0
    while attempts < cfg.max_retries:
        attempts += 1
        payload = {"model": cfg.deepseek_model, "messages": messages, "max_tokens": 1400, "stream": False,
                   "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"}, "temperature": 0.0}
        try:
            data, latency = post_json(endpoint(cfg.deepseek_base), cfg.deepseek_key, payload)
            total_latency += latency; inp1, out1 = usage_from(data); inp += inp1; out += out1
            last_raw = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            value = parse_json_response(last_raw); validate_payload(value, TEXT_VALIDATOR, str(item["id"]))
            return {"status": "success", "payload": value, "raw": last_raw, "attempts": attempts,
                    "retries": retries, "input_tokens": inp, "output_tokens": out, "latency": total_latency}
        except Exception as exc:
            retries += 1
            if attempts >= cfg.max_retries:
                return {"status": "failed", "payload": None, "raw": last_raw, "attempts": attempts,
                        "retries": retries - 1, "input_tokens": inp, "output_tokens": out,
                        "latency": total_latency, "error": f"{type(exc).__name__}: {exc}"}
            messages.append({"role": "assistant", "content": last_raw or "{}"})
            messages.append({"role": "user", "content": repair_instruction(last_raw, [str(exc)], "DeepSeek预筛")})
            time.sleep(min(8.0, 2 ** (attempts - 1)))
    raise AssertionError("unreachable")


def enforce_vision_policy(value: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(value); scores = result["scores"]
    core_min = all(int(scores[name]) >= 2 for name in (
        "dynamic_process", "explicit_rule", "initial_condition", "outcome_verifiable",
        "engine_feasibility", "educational_value", "data_quality",
    ))
    if result["final_category"] == "A_CORE" and not core_min:
        result["final_category"] = "B_EXTENSION" if int(scores["dynamic_process"]) >= 2 and int(scores["outcome_verifiable"]) >= 2 else "C_STATIC"
        result["manual_review_required"] = True
        result["selection_reason"] += " [policy: model A_CORE failed non-compensatory minima]"
    if result["final_category"] in {"A_CORE", "B_EXTENSION"} and result["simulation_family"] in {"static_diagram", "none"}:
        result["manual_review_required"] = True
    return result


def call_qwen(cfg: Config, item: Mapping[str, Any]) -> Dict[str, Any]:
    data_bytes, media, meta = decode_image(item.get("image"))
    if not data_bytes:
        return {"status": "failed", "payload": None, "raw": "", "attempts": 0, "retries": 0,
                "input_tokens": 0, "output_tokens": 0, "latency": 0.0,
                "error": "image_error: no decodable real image for multimodal review"}
    data_url = f"data:{media or 'image/png'};base64," + base64.b64encode(data_bytes).decode("ascii")
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": vision_prompt(item)},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]
    attempts = 0; retries = 0; total_latency = 0.0; last_raw = ""; inp = out = 0
    while attempts < cfg.max_retries:
        attempts += 1
        request_payload = {"model": cfg.qwen_model, "messages": messages, "max_tokens": 2400,
                           "temperature": 0.1, "stream": False}
        try:
            response, latency = post_json(endpoint(cfg.qwen_base), cfg.qwen_key, request_payload)
            total_latency += latency; inp1, out1 = usage_from(response); inp += inp1; out += out1
            last_raw = str(((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            value = parse_json_response(last_raw); validate_payload(value, VISION_VALIDATOR, str(item["id"]))
            value = enforce_vision_policy(value)
            return {"status": "success", "payload": value, "raw": last_raw, "attempts": attempts,
                    "retries": retries, "input_tokens": inp, "output_tokens": out, "latency": total_latency}
        except Exception as exc:
            retries += 1
            if attempts >= cfg.max_retries:
                return {"status": "failed", "payload": None, "raw": last_raw, "attempts": attempts,
                        "retries": retries - 1, "input_tokens": inp, "output_tokens": out,
                        "latency": total_latency, "error": f"{type(exc).__name__}: {exc}"}
            # Keep the original image in the first user message; only add textual repair context.
            messages.append({"role": "assistant", "content": last_raw or "{}"})
            messages.append({"role": "user", "content": repair_instruction(last_raw, [str(exc)], "Qwen终筛")})
            time.sleep(min(8.0, 2 ** (attempts - 1)))
    raise AssertionError("unreachable")


def source_records(cfg: Config, only_ids: Optional[set[str]] = None) -> Iterator[Dict[str, Any]]:
    seen_raw_ids: set[str] = set()
    for line_number, raw, error in iter_source(cfg.source):
        if raw is None: continue
        item = normalize_record(raw, line_number)
        # Duplicate-ID suffixes mirror audit_dataset even when the original ID
        # is itself part of ``only_ids``.
        raw_id = item["id"]
        if raw_id in seen_raw_ids:
            item["id"] = f"{raw_id}__duplicate_id_line_{line_number}"
        else:
            seen_raw_ids.add(raw_id)
        if only_ids is not None and item["id"] not in only_ids:
            continue
        yield item


def run_parallel_stage(cfg: Config, db: Checkpoint, stage: str, ids: set[str], worker_count: int,
                       function: Any) -> Dict[str, int]:
    completed = db.completed_ids(stage)
    pending_ids = ids - completed
    stats = Counter(); stats["cached"] = len(ids & completed)
    if not pending_ids:
        return dict(stats)
    print(f"[{stage}] pending={len(pending_ids)} cached={len(completed)} workers={worker_count}", flush=True)
    def record_result(item: Mapping[str, Any], future: futures.Future) -> None:
        item_id = str(item["id"])
        try: result = future.result()
        except Exception as exc:
            result = {"status": "failed", "payload": None, "raw": "", "attempts": 1,
                      "retries": 0, "input_tokens": 0, "output_tokens": 0, "latency": 0.0,
                      "error": f"{type(exc).__name__}: {exc}"}
        db.put_stage(stage, item_id, result["status"], result.get("payload"), result.get("raw", ""),
                     int(result.get("attempts", 0)), int(result.get("input_tokens", 0)),
                     int(result.get("output_tokens", 0)), float(result.get("latency", 0.0)))
        stats["calls"] += int(result.get("attempts", 0)); stats["retries"] += int(result.get("retries", 0))
        stats["input_tokens"] += int(result.get("input_tokens", 0)); stats["output_tokens"] += int(result.get("output_tokens", 0))
        if result["status"] == "success": stats["success"] += 1
        else:
            stats["failed"] += 1
            db.error(item_id, stage, "api_or_schema_error", result.get("error", "unknown"), int(result.get("attempts", 0)))
        done = stats["success"] + stats["failed"]
        if done % 25 == 0:
            print(f"[{stage}] processed={done}/{len(pending_ids)} success={stats['success']} failed={stats['failed']}", flush=True)

    # Keep one executor alive for the complete stage so worker-local HTTP
    # sessions reuse TLS connections. Bound in-flight work to avoid retaining
    # the full image dataset in memory.
    source = iter(source_records(cfg, pending_ids))
    exhausted = False
    in_flight: Dict[futures.Future, Dict[str, Any]] = {}
    with futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
        while not exhausted or in_flight:
            while not exhausted and len(in_flight) < worker_count * 3:
                try:
                    item = next(source)
                except StopIteration:
                    exhausted = True
                    break
                if item["id"] in pending_ids:
                    in_flight[pool.submit(function, cfg, item)] = item
            if not in_flight:
                continue
            done, _ = futures.wait(in_flight, return_when=futures.FIRST_COMPLETED)
            for future in done:
                item = in_flight.pop(future)
                record_result(item, future)
    return dict(stats)


def eligible_ids(db: Checkpoint) -> set[str]:
    return {item_id for item_id, q in db.quality_map().items() if not q.get("hard_reject")}


def select_smoke_ids(cfg: Config, db: Checkpoint) -> List[str]:
    existing = db.get_meta("smoke_ids")
    if existing: return list(existing)
    buckets: Dict[str, List[str]] = defaultdict(list)
    for item_id, q in db.quality_map().items():
        if not q.get("hard_reject") and q.get("image", {}).get("decodable"):
            buckets[str(q.get("subject_base"))].append(item_id)
    chosen: List[str] = []
    for subject in sorted(buckets):
        ids = sorted(buckets[subject], key=lambda x: stable_hash(f"{cfg.seed}:{x}"))
        if ids and len(chosen) < 5: chosen.append(ids[0])
    if len(chosen) < 5:
        remaining = sorted(eligible_ids(db) - set(chosen), key=lambda x: stable_hash(f"smoke:{cfg.seed}:{x}"))
        chosen.extend(remaining[:5-len(chosen)])
    db.set_meta("smoke_ids", chosen)
    return chosen


def smoke_test(cfg: Config, db: Checkpoint) -> None:
    ids = set(select_smoke_ids(cfg, db)); print("[smoke] ids=" + ",".join(sorted(ids)), flush=True)
    t = run_parallel_stage(cfg, db, "deepseek", ids, min(5, cfg.deepseek_workers), call_deepseek)
    if db.completed_ids("deepseek") & ids != ids:
        raise RuntimeError("DeepSeek 5-sample smoke test did not complete successfully")
    q = run_parallel_stage(cfg, db, "qwen", ids, min(5, cfg.qwen_workers), call_qwen)
    if db.completed_ids("qwen") & ids != ids:
        raise RuntimeError("Qwen 5-sample smoke test did not complete successfully")
    db.set_meta("smoke_complete", {"ids": sorted(ids), "deepseek": t, "qwen": q, "completed_at": utc_now()})


def route_initial_qwen(cfg: Config, db: Checkpoint) -> Tuple[set[str], Dict[str, Any]]:
    quality = db.quality_map(); text = db.stage_map("deepseek")
    routes: set[str] = set(); high_static: List[str] = []
    for item_id, q in quality.items():
        if q.get("hard_reject"): continue
        state = text.get(item_id)
        if not state or state["status"] != "success": continue
        value = state["payload"]; label = value["prefilter_label"]; confidence = float(value["confidence"])
        conflict = any("冲突" in str(x) or "conflict" in str(x).lower() for x in value.get("missing_information", []))
        if label in {"likely_dynamic", "uncertain"} or value["image_needed"] or confidence < 0.85 or conflict:
            routes.add(item_id)
        elif label in {"likely_static", "obvious_reject"}:
            high_static.append(item_id)
    # Union of >=10% samples independently stratified by subject, grade, and primary knowledge point.
    sampled: set[str] = set(); strata_members: Dict[str, List[str]] = defaultdict(list)
    for item_id in high_static:
        q = quality[item_id]
        keys = (
            f"subject:{q.get('subject_base','unknown')}", f"grade:{q.get('grade','unknown')}",
            f"knowledge:{(q.get('knowledge_points') or ['unknown'])[0]}",
        )
        for key in keys: strata_members[key].append(item_id)
    sample_by_stratum: Dict[str, List[str]] = {}
    for key, members in strata_members.items():
        ordered = sorted(set(members), key=lambda x: stable_hash(f"audit:{cfg.seed}:{key}:{x}"))
        n = max(1, math.ceil(len(ordered) * 0.10)); selected = ordered[:n]
        sample_by_stratum[key] = selected; sampled.update(selected)
    routes.update(sampled)
    audit = {"seed": cfg.seed, "rate": 0.10, "sample_by_stratum": sample_by_stratum,
             "sample_ids": sorted(sampled), "initial_route_ids": sorted(routes)}
    db.set_meta("qwen_initial_routing", audit)
    (cfg.output / "qwen_audit_sampling.json").write_text(json_dumps(audit, pretty=True) + "\n", encoding="utf-8")
    return routes, audit


def route_expansion(cfg: Config, db: Checkpoint, audit: Mapping[str, Any]) -> set[str]:
    quality = db.quality_map(); qwen = db.stage_map("qwen")
    expansion: set[str] = set(); triggered: Dict[str, Any] = {}
    for key, sample_ids in (audit.get("sample_by_stratum") or {}).items():
        completed = [qwen.get(item_id) for item_id in sample_ids]
        completed = [state for state in completed if state and state["status"] == "success"]
        if not completed: continue
        positives = sum(state["payload"]["final_category"] in {"A_CORE", "B_EXTENSION"} for state in completed)
        rate = positives / len(completed)
        if rate > 0.02:
            kind, value = key.split(":", 1)
            for item_id, q in quality.items():
                if q.get("hard_reject"): continue
                match = (
                    (kind == "subject" and str(q.get("subject_base")) == value) or
                    (kind == "grade" and str(q.get("grade")) == value) or
                    (kind == "knowledge" and str((q.get("knowledge_points") or ["unknown"])[0]) == value)
                )
                if match: expansion.add(item_id)
            triggered[key] = {"audited": len(completed), "positive": positives, "rate": rate}
    meta = {"threshold": 0.02, "triggered_strata": triggered, "expansion_ids": sorted(expansion), "completed_at": utc_now()}
    db.set_meta("qwen_expansion", meta)
    (cfg.output / "qwen_audit_expansion.json").write_text(json_dumps(meta, pretty=True) + "\n", encoding="utf-8")
    return expansion


def full_screen(cfg: Config, db: Checkpoint) -> None:
    ids = eligible_ids(db)
    deep_stats = run_parallel_stage(cfg, db, "deepseek", ids, cfg.deepseek_workers, call_deepseek)
    print("[deepseek] stage summary " + json_dumps(deep_stats), flush=True)
    initial, audit = route_initial_qwen(cfg, db)
    qwen_stats = run_parallel_stage(cfg, db, "qwen", initial, cfg.qwen_workers, call_qwen)
    print("[qwen initial] stage summary " + json_dumps(qwen_stats), flush=True)
    expansion = route_expansion(cfg, db, audit)
    expansion_stats = run_parallel_stage(cfg, db, "qwen", expansion, cfg.qwen_workers, call_qwen)
    print("[qwen expansion] stage summary " + json_dumps(expansion_stats), flush=True)
    db.set_meta("screen_complete", {"deepseek": deep_stats, "qwen_initial": qwen_stats, "qwen_expansion": expansion_stats, "completed_at": utc_now()})


def base_final_record(item_id: str, quality: Mapping[str, Any], text: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    pre = text["payload"] if text and text.get("status") == "success" else None
    return {
        "id": item_id, "source_file": quality.get("source_file", SOURCE_RELATIVE), "subject": quality.get("subject"),
        "grade": quality.get("grade"), "question_type": quality.get("question_type"),
        "knowledge_points": quality.get("knowledge_points") or [],
        "prefilter_label": pre.get("prefilter_label") if pre else "uncertain",
        "prefilter_confidence": float(pre.get("confidence", 0.0)) if pre else 0.0,
        "final_category": None, "decision_source": "data_quality", "simulation_family": "none",
        "recommended_backend": "none", "scores": {name: 0 for name in SCORE_NAMES},
        "required_objects": [], "initial_state": [], "governing_rules": [], "dynamic_description": "",
        "expected_observable": "", "answer_verification": "", "missing_information": [],
        "reasonable_defaults": [], "conflict_flag": False, "conflict_description": "",
        "selection_reason": "", "manual_review_required": False, "confidence": 0.0,
    }


def make_final_records(cfg: Config, db: Checkpoint) -> List[Dict[str, Any]]:
    quality = db.quality_map(); text = db.stage_map("deepseek"); qwen = db.stage_map("qwen")
    records: List[Dict[str, Any]] = []
    for item_id, q in sorted(quality.items(), key=lambda kv: int(kv[1].get("line_number", 0))):
        record = base_final_record(item_id, q, text.get(item_id)); record["scores"]["data_quality"] = int(q.get("data_quality_score", 0))
        if q.get("hard_reject"):
            record.update({"final_category": "D_REJECT", "decision_source": "data_quality",
                           "selection_reason": "; ".join(q.get("hard_reject_reasons") or []), "confidence": 1.0})
        elif qwen.get(item_id, {}).get("status") == "success":
            value = qwen[item_id]["payload"]
            for key in ("final_category", "simulation_family", "recommended_backend", "scores", "required_objects",
                        "initial_state", "governing_rules", "dynamic_description", "expected_observable", "answer_verification",
                        "missing_information", "reasonable_defaults", "conflict_flag", "conflict_description",
                        "selection_reason", "manual_review_required", "confidence"):
                record[key] = value[key]
            record["decision_source"] = "qwen_multimodal"
        elif qwen.get(item_id, {}).get("status") == "failed":
            record.update({"final_category": None, "decision_source": "qwen_multimodal",
                           "manual_review_required": True, "selection_reason": "Qwen API/schema/image failure; not auto-rejected"})
        elif text.get(item_id, {}).get("status") == "success":
            pre = text[item_id]["payload"]
            if pre["prefilter_label"] == "likely_static" and float(pre["confidence"]) >= 0.85:
                record.update({"final_category": "C_STATIC", "decision_source": "deepseek_text",
                               "simulation_family": "static_diagram", "recommended_backend": "svg",
                               "selection_reason": pre["reason"], "confidence": float(pre["confidence"])})
            elif pre["prefilter_label"] == "obvious_reject" and float(pre["confidence"]) >= 0.85:
                record.update({"final_category": "D_REJECT", "decision_source": "deepseek_text",
                               "selection_reason": pre["reason"], "confidence": float(pre["confidence"])})
            else:
                record.update({"final_category": None, "decision_source": "deepseek_text", "manual_review_required": True,
                               "selection_reason": "required multimodal review was not completed"})
        else:
            record.update({"final_category": None, "decision_source": "deepseek_text", "manual_review_required": True,
                           "selection_reason": "DeepSeek API/schema failure; not auto-rejected"})
        records.append(record)
    return records


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json_dumps(row) + "\n")
    os.replace(tmp, path)


def export_stage_files(cfg: Config, db: Checkpoint) -> None:
    quality = db.quality_map(); text = db.stage_map("deepseek"); qwen = db.stage_map("qwen")
    write_jsonl(cfg.output / "data_quality_results.jsonl", (quality[k] for k in sorted(quality, key=lambda x: quality[x].get("line_number", 0))))
    write_jsonl(cfg.output / "deepseek_prefilter_results.jsonl", (
        {"id": k, "status": v["status"], "model": cfg.deepseek_model, "prompt_version": TEXT_PROMPT_VERSION,
         "attempts": v["attempts"], "input_tokens": v["input_tokens"], "output_tokens": v["output_tokens"],
         "latency_seconds": v["latency"], "result": v["payload"]} for k, v in sorted(text.items())
    ))
    write_jsonl(cfg.output / "qwen_multimodal_results.jsonl", (
        {"id": k, "status": v["status"], "model": cfg.qwen_model, "prompt_version": VISION_PROMPT_VERSION,
         "attempts": v["attempts"], "input_tokens": v["input_tokens"], "output_tokens": v["output_tokens"],
         "latency_seconds": v["latency"], "result": v["payload"]} for k, v in sorted(qwen.items())
    ))
    conn = db.connection()
    write_jsonl(cfg.output / "error_log.jsonl", (
        {"id": row[0], "stage": row[1], "error_type": row[2], "message": row[3], "attempt": row[4], "timestamp": row[5]}
        for row in conn.execute("SELECT id,stage,error_type,message,attempt,timestamp FROM errors ORDER BY seq")
    ))
    audit_dir = cfg.output / "audit"; audit_dir.mkdir(exist_ok=True)
    for stage in ("deepseek", "qwen"):
        write_jsonl(audit_dir / f"{stage}_raw_responses.jsonl", (
            {"id": row[0], "status": row[1], "raw_response": row[2], "attempts": row[3], "updated_at": row[4]}
            for row in conn.execute("SELECT id,status,raw_response,attempts,updated_at FROM stage WHERE stage=? ORDER BY id", (stage,))
        ))


def thumbnail_data(raw: Any, max_side: int = 360) -> str:
    data, _, _ = decode_image(raw)
    if not data: return ""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB"); image.thumbnail((max_side, max_side))
            buffer = io.BytesIO(); image.save(buffer, "JPEG", quality=82, optimize=True)
            return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception: return ""


REVIEW_PAGE_SIZE = 100


REVIEW_STYLE = """
:root{color-scheme:light;--bg:#f4f6f8;--card:#fff;--ink:#17212b;--muted:#667085;--line:#d9dee7;--accent:#1769aa;--good:#147d64;--warn:#a15c00;--bad:#b42318}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:var(--accent)}header,.toolbar,.page-nav{position:sticky;z-index:5;background:rgba(255,255,255,.96);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
header{top:0;padding:12px 20px;display:flex;gap:16px;align-items:center;justify-content:space-between}header h1{font-size:18px;margin:0}.header-actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button,.button,select,input,textarea{font:inherit;border:1px solid #b8c0cc;border-radius:7px;background:#fff;padding:7px 9px}.button,button{cursor:pointer;text-decoration:none}.button.primary,button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
main{max-width:1500px;margin:0 auto;padding:18px}.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px}.metric{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px}.metric b{display:block;font-size:22px}.metric span{color:var(--muted)}
.filters{display:grid;grid-template-columns:minmax(240px,2fr) repeat(4,minmax(130px,1fr));gap:8px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px}.filters input,.filters select{width:100%}.result-bar{display:flex;justify-content:space-between;align-items:center;margin:12px 0;gap:12px}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line)}th,td{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}th{background:#eef2f6;position:sticky;top:57px}.id{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;word-break:break-all}.snippet{max-width:520px;color:#475467}.list-image-cell{width:170px}.list-thumb{display:block;width:160px;height:112px;object-fit:contain;border:1px solid var(--line);border-radius:7px;background:#fff}.pager{display:flex;gap:6px;align-items:center;justify-content:center;margin:15px 0}.pager button[disabled]{opacity:.45;cursor:not-allowed}
.page-nav{top:57px;padding:8px 20px;display:flex;justify-content:space-between;align-items:center}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(470px,1fr));gap:16px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;box-shadow:0 2px 8px rgba(16,24,40,.06);scroll-margin-top:110px}.card h2{font-size:13px;margin:0 0 8px;word-break:break-all}.badges{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}.badge{display:inline-block;border-radius:999px;padding:3px 8px;background:#e9eef5;font-size:12px}.badge.core{background:#dff5ed;color:#116149}.badge.extension{background:#e6efff;color:#194ea6}.badge.manual{background:#fff1d6;color:#8b5200}.badge.reject{background:#fee4e2;color:#912018}
.problem-image{display:block;max-width:100%;max-height:620px;margin:8px auto;border:1px solid var(--line);border-radius:8px;background:#fff}.question{white-space:pre-wrap;font-size:15px}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto}.panel h3{font-size:13px;margin:0 0 6px}.panel pre,details pre{white-space:pre-wrap;word-break:break-word;font-size:11px;max-height:360px;overflow:auto}.score-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:4px;font-size:12px}.review-box{margin-top:12px;border-top:2px solid #d9e5f2;padding-top:10px}.review-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.review-box textarea{width:100%;min-height:72px;margin-top:8px}.saved{color:var(--good);font-weight:600;margin-left:8px}.muted{color:var(--muted)}details{margin-top:8px}details summary{cursor:pointer;font-weight:600}
@media(max-width:850px){.filters,.two-col,.review-grid{grid-template-columns:1fr}.cards{grid-template-columns:1fr}header,.page-nav{position:static}th{position:static}.list-image-cell{width:112px}.list-thumb{width:104px;height:84px}}
"""


REVIEW_SCRIPT = r"""
(function(){
  const PREFIX='k12vista-review:';
  const reviewerKey=PREFIX+'reviewer';
  const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function reviewer(){const node=document.querySelector('[data-reviewer]');return (node&&node.value.trim())||localStorage.getItem(reviewerKey)||'';}
  function setReviewer(value){localStorage.setItem(reviewerKey,value);document.querySelectorAll('[data-reviewer]').forEach(n=>{if(n.value!==value)n.value=value;});}
  function itemKey(id){return PREFIX+'item:'+id;}
  function read(id){try{return JSON.parse(localStorage.getItem(itemKey(id))||'{}')}catch(_){return {}}}
  function write(card){
    const id=card.dataset.itemId;const value={id,reviewer_id:reviewer(),auto_category:card.dataset.autoCategory||'',human_category:card.querySelector('[data-human-category]').value,accept_auto:card.querySelector('[data-accept-auto]').checked,notes:card.querySelector('[data-notes]').value,review_time:new Date().toISOString()};
    setReviewer(value.reviewer_id);localStorage.setItem(itemKey(id),JSON.stringify(value));const status=card.querySelector('[data-save-status]');if(status){status.textContent='已保存';setTimeout(()=>status.textContent='',1800)}
  }
  function hydrate(){
    const savedReviewer=localStorage.getItem(reviewerKey)||'';document.querySelectorAll('[data-reviewer]').forEach(n=>{n.value=savedReviewer;n.addEventListener('change',()=>setReviewer(n.value.trim()))});
    document.querySelectorAll('[data-review-card]').forEach(card=>{const value=read(card.dataset.itemId);if(value.human_category)card.querySelector('[data-human-category]').value=value.human_category;if(value.accept_auto)card.querySelector('[data-accept-auto]').checked=true;if(value.notes)card.querySelector('[data-notes]').value=value.notes;card.querySelector('[data-save]').addEventListener('click',()=>write(card));});
    document.querySelectorAll('[data-export]').forEach(n=>n.addEventListener('click',exportCsv));
  }
  function csvCell(v){const s=String(v??'');return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
  function exportCsv(){
    const rows=[];for(let i=0;i<localStorage.length;i++){const key=localStorage.key(i);if(key&&key.startsWith(PREFIX+'item:')){try{rows.push(JSON.parse(localStorage.getItem(key)))}catch(_){}}}
    rows.sort((a,b)=>a.id.localeCompare(b.id));const fields=['id','reviewer_id','auto_category','human_category','accept_auto','notes','review_time'];const text='\ufeff'+[fields.join(','),...rows.map(r=>fields.map(f=>csvCell(r[f])).join(','))].join('\n');const blob=new Blob([text],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='k12vista_human_annotations.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
  }
  function initIndex(){
    if(!window.REVIEW_ITEMS)return;const items=window.REVIEW_ITEMS;const filters={q:document.querySelector('#filter-q'),category:document.querySelector('#filter-category'),subject:document.querySelector('#filter-subject'),grade:document.querySelector('#filter-grade'),manual:document.querySelector('#filter-manual')};
    [...new Set(items.map(x=>x.subject))].sort().forEach(v=>filters.subject.insertAdjacentHTML('beforeend','<option value="'+esc(v)+'">'+esc(v)+'</option>'));[...new Set(items.map(x=>x.grade))].sort().forEach(v=>filters.grade.insertAdjacentHTML('beforeend','<option value="'+esc(v)+'">'+esc(v)+'</option>'));
    let current=1;const pageSize=100;function selected(){const q=filters.q.value.trim().toLowerCase();return items.filter(x=>(!q||(x.id+' '+x.snippet+' '+x.knowledge).toLowerCase().includes(q))&&(!filters.category.value||x.category===filters.category.value)&&(!filters.subject.value||x.subject===filters.subject.value)&&(!filters.grade.value||x.grade===filters.grade.value)&&(!filters.manual.value||(filters.manual.value==='yes')===x.manual));}
    function detailUrl(x){return 'pages/page_'+String(x.page).padStart(3,'0')+'.html#'+esc(x.anchor)}
    function render(){const rows=selected();const pages=Math.max(1,Math.ceil(rows.length/pageSize));current=Math.min(current,pages);const start=(current-1)*pageSize;document.querySelector('#result-count').textContent='匹配 '+rows.length+' 题；第 '+current+'/'+pages+' 页';document.querySelector('#result-body').innerHTML=rows.slice(start,start+pageSize).map(x=>{const url=detailUrl(x);const picture=x.image?'<a href="'+url+'"><img class="list-thumb" loading="lazy" decoding="async" src="images/'+esc(x.image)+'" alt="题图"></a>':'<span class="muted">无题图</span>';return '<tr><td class="list-image-cell">'+picture+'</td><td><span class="badge '+(x.category==='A_CORE'?'core':x.category==='B_EXTENSION'?'extension':x.category==='D_REJECT'?'reject':'')+'">'+esc(x.category)+'</span></td><td>'+esc(x.subject)+'</td><td>'+esc(x.grade)+'</td><td>'+esc(x.simulation_family)+'</td><td class="snippet">'+esc(x.snippet)+'</td><td class="id">'+esc(x.id)+'</td><td><a href="'+url+'">打开</a></td></tr>'}).join('');document.querySelector('#prev').disabled=current<=1;document.querySelector('#next').disabled=current>=pages;}
    Object.values(filters).forEach(n=>n.addEventListener(n===filters.q?'input':'change',()=>{current=1;render()}));document.querySelector('#prev').addEventListener('click',()=>{current--;render()});document.querySelector('#next').addEventListener('click',()=>{current++;render()});render();
  }
  document.addEventListener('DOMContentLoaded',()=>{hydrate();initIndex()});
})();
"""


def _read_jsonl_ids(path: Path) -> List[str]:
    if not path.exists(): return []
    ids: List[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line); item_id = str(value.get("id") or "")
                if item_id: ids.append(item_id)
            except (json.JSONDecodeError, AttributeError):
                continue
    return ids


def _image_extension(media: str, data: bytes) -> str:
    known = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}
    if media in known: return known[media]
    try:
        with Image.open(io.BytesIO(data)) as image:
            return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}.get(str(image.format), ".img")
    except Exception:
        return ".img"


def _review_card(item_id: str, result: Mapping[str, Any], item: Mapping[str, Any], image_name: str) -> str:
    category = str(result.get("final_category") or "MANUAL_REVIEW")
    badge_class = "core" if category == "A_CORE" else "extension" if category == "B_EXTENSION" else "reject" if category == "D_REJECT" else ""
    badges = [f"<span class='badge {badge_class}'>{html.escape(category)}</span>"]
    if result.get("manual_review_required"): badges.append("<span class='badge manual'>需人工复核</span>")
    if result.get("conflict_flag"): badges.append("<span class='badge reject'>存在冲突</span>")
    scores = result.get("scores") or {}
    score_html = "".join(f"<div>{html.escape(name)}: <b>{html.escape(str(scores.get(name,'')))}</b></div>" for name in SCORE_NAMES)
    image_html = (f"<a href='../images/{html.escape(image_name)}' target='_blank'><img class='problem-image' loading='lazy' src='../images/{html.escape(image_name)}' alt='题图'></a>" if image_name else "<p class='muted'>无可用题图</p>")
    anchor = "item-" + stable_hash(item_id)[:16]
    return f"""<article class='card' id='{anchor}' data-review-card data-item-id='{html.escape(item_id)}' data-auto-category='{html.escape(category)}'>
<h2>{html.escape(item_id)}</h2><div class='badges'>{''.join(badges)}<span class='badge'>{html.escape(str(result.get('subject','')))}</span><span class='badge'>{html.escape(str(result.get('question_type','')))}</span></div>
{image_html}<div class='question'>{html.escape(str(item.get('question','')))}</div>
<div class='two-col'><section class='panel'><h3>自动判断</h3><p><b>仿真类型：</b>{html.escape(str(result.get('simulation_family','')))}<br><b>后端：</b>{html.escape(str(result.get('recommended_backend','')))}<br><b>置信度：</b>{html.escape(str(result.get('confidence','')))}</p><p>{html.escape(str(result.get('selection_reason','')))}</p><div class='score-grid'>{score_html}</div></section>
<section class='panel'><h3>教学与仿真信息</h3><p><b>动态过程：</b>{html.escape(str(result.get('dynamic_description','')))}</p><p><b>预期观测：</b>{html.escape(str(result.get('expected_observable','')))}</p><p><b>答案验证：</b>{html.escape(str(result.get('answer_verification','')))}</p><p><b>缺失信息：</b>{html.escape('；'.join(result.get('missing_information') or []))}</p></section></div>
<details><summary>标准答案与解析</summary><div class='two-col'><section class='panel'><h3>答案</h3><pre>{html.escape(json_dumps(item.get('ground_truth') or [], pretty=True))}</pre></section><section class='panel'><h3>解析</h3><pre>{html.escape(json_dumps(item.get('solution') or [], pretty=True))}</pre></section></div></details>
<details><summary>完整筛选 JSON</summary><pre>{html.escape(json_dumps(result, pretty=True))}</pre></details>
<section class='review-box'><h3>人工标注</h3><div class='review-grid'><label>人工类别<select data-human-category><option value=''>未选择</option><option>A_CORE</option><option>B_EXTENSION</option><option>C_STATIC</option><option>D_REJECT</option><option>MANUAL_PENDING</option></select></label><label><input type='checkbox' data-accept-auto> 接受自动分类</label></div><textarea data-notes placeholder='记录动态过程、规律、参数充分性、冲突或修改理由'></textarea><div><button class='primary' data-save>保存到浏览器</button><span class='saved' data-save-status></span></div></section></article>"""


def generate_gallery(cfg: Config, records: Sequence[Mapping[str, Any]]) -> None:
    """Generate a paginated, offline review tool without inline Base64 images."""
    by_id = {str(r["id"]): r for r in records}
    ordered: List[str] = []
    ordered.extend(_read_jsonl_ids(cfg.output / "manual_review_priority.jsonl"))
    ordered.extend(_read_jsonl_ids(cfg.output / "first_round_recommendations.jsonl"))
    ordered.extend(str(r["id"]) for r in records if r.get("final_category") in {"A_CORE", "B_EXTENSION"})
    ordered.extend(str(r["id"]) for r in records if r.get("manual_review_required"))
    for category in ("C_STATIC", "D_REJECT"):
        candidates = [str(r["id"]) for r in records if r.get("final_category") == category]
        candidates.sort(key=lambda x: stable_hash(f"gallery:{cfg.seed}:{x}")); ordered.extend(candidates[:100])
    # Preserve review priority while eliminating prior duplicate cards.
    seen: set[str] = set(); priorities = [item_id for item_id in ordered if item_id in by_id and not (item_id in seen or seen.add(item_id))]
    print(f"[review] unique_items={len(priorities)} pages={max(1, math.ceil(len(priorities) / REVIEW_PAGE_SIZE))}", flush=True)

    review_root = cfg.output / "review"
    if review_root.parent != cfg.output: raise RuntimeError("unsafe review output path")
    if review_root.exists(): shutil.rmtree(review_root)
    pages_dir = review_root / "pages"; images_dir = review_root / "images"; assets_dir = review_root / "assets"
    for directory in (pages_dir, images_dir, assets_dir): directory.mkdir(parents=True, exist_ok=True)
    (assets_dir / "style.css").write_text(REVIEW_STYLE, encoding="utf-8")
    (assets_dir / "review.js").write_text(REVIEW_SCRIPT, encoding="utf-8")

    wanted = set(priorities); source: Dict[str, Dict[str, Any]] = {}; image_names: Dict[str, str] = {}
    for item in source_records(cfg, wanted):
        item_id = str(item["id"]); data, media, _ = decode_image(item.get("image"))
        if data:
            filename = stable_hash(item_id) + _image_extension(media, data)
            (images_dir / filename).write_bytes(data); image_names[item_id] = filename
        item = dict(item); item.pop("image", None); source[item_id] = item
        if len(source) % 1000 == 0: print(f"[review] extracted={len(source)}/{len(priorities)} images={len(image_names)}", flush=True)

    page_count = max(1, math.ceil(len(priorities) / REVIEW_PAGE_SIZE)); search_items: List[Dict[str, Any]] = []
    for page_number in range(1, page_count + 1):
        page_ids = priorities[(page_number - 1) * REVIEW_PAGE_SIZE:page_number * REVIEW_PAGE_SIZE]
        cards: List[str] = []
        for item_id in page_ids:
            result = by_id[item_id]; item = source.get(item_id, {})
            cards.append(_review_card(item_id, result, item, image_names.get(item_id, "")))
            search_items.append({
                "id": item_id, "category": str(result.get("final_category") or "MANUAL_REVIEW"),
                "subject": str(result.get("subject") or "unknown"), "grade": str(result.get("grade") or "unknown"),
                "question_type": str(result.get("question_type") or ""), "simulation_family": str(result.get("simulation_family") or "none"),
                "backend": str(result.get("recommended_backend") or "none"), "manual": bool(result.get("manual_review_required")),
                "confidence": float(result.get("confidence") or 0), "snippet": str(item.get("question") or "")[:220],
                "knowledge": "；".join(result.get("knowledge_points") or []), "page": page_number,
                "image": image_names.get(item_id, ""),
                "anchor": "item-" + stable_hash(item_id)[:16],
            })
        previous_link = f"page_{page_number-1:03d}.html" if page_number > 1 else ""
        next_link = f"page_{page_number+1:03d}.html" if page_number < page_count else ""
        navigation = ((f"<a class='button' href='{previous_link}'>← 上一页</a>" if previous_link else "<span></span>")
                      + f"<span>第 {page_number}/{page_count} 页 · 本页 {len(page_ids)} 题</span>"
                      + (f"<a class='button' href='{next_link}'>下一页 →</a>" if next_link else "<span></span>"))
        page_html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>K12Vista复核 第{page_number}页</title><link rel='stylesheet' href='../assets/style.css'></head><body>
<header><h1>K12Vista 分页人工复核</h1><div class='header-actions'><label>审核者 <input data-reviewer placeholder='姓名或编号'></label><button data-export>导出已保存CSV</button><a class='button primary' href='../index.html'>搜索与筛选</a></div></header><nav class='page-nav'>{navigation}</nav><main><div class='cards'>{''.join(cards)}</div><div class='pager'>{navigation}</div></main><script src='../assets/review.js'></script></body></html>"""
        (pages_dir / f"page_{page_number:03d}.html").write_text(page_html, encoding="utf-8")
        if page_number % 25 == 0 or page_number == page_count: print(f"[review] pages={page_number}/{page_count}", flush=True)

    js_payload = json_dumps(search_items).replace("</", "<\\/")
    (assets_dir / "search-index.js").write_text("window.REVIEW_ITEMS=" + js_payload + ";\n", encoding="utf-8")
    category_counts = Counter(item["category"] for item in search_items)
    index_html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>K12Vista分页复核</title><link rel='stylesheet' href='assets/style.css'></head><body>
<header><h1>K12Vista 分页人工复核</h1><div class='header-actions'><label>审核者 <input data-reviewer placeholder='姓名或编号'></label><button data-export>导出已保存CSV</button><a class='button' href='../summary.md'>筛选摘要</a></div></header><main>
<section class='summary'><div class='metric'><b>{len(search_items)}</b><span>唯一复核题目</span></div><div class='metric'><b>{page_count}</b><span>分页，每页≤{REVIEW_PAGE_SIZE}</span></div><div class='metric'><b>{category_counts.get('A_CORE',0)}</b><span>A_CORE</span></div><div class='metric'><b>{category_counts.get('B_EXTENSION',0)}</b><span>B_EXTENSION</span></div><div class='metric'><b>{sum(x['manual'] for x in search_items)}</b><span>需人工复核</span></div></section>
<section class='filters'><input id='filter-q' placeholder='搜索ID、题目文字或知识点'><select id='filter-category'><option value=''>全部类别</option><option>A_CORE</option><option>B_EXTENSION</option><option>C_STATIC</option><option>D_REJECT</option><option>MANUAL_REVIEW</option></select><select id='filter-subject'><option value=''>全部学科</option></select><select id='filter-grade'><option value=''>全部年级</option></select><select id='filter-manual'><option value=''>全部复核状态</option><option value='yes'>仅需人工复核</option><option value='no'>仅无需复核</option></select></section>
<div class='result-bar'><b id='result-count'></b><span class='muted'>每页最多懒加载100张题图；人工标注保存在当前浏览器 localStorage，请定期导出CSV。</span></div><table><thead><tr><th>题图</th><th>类别</th><th>学科</th><th>年级</th><th>仿真类型</th><th>题目摘要</th><th>ID</th><th>页面</th></tr></thead><tbody id='result-body'></tbody></table><div class='pager'><button id='prev'>← 上一页</button><button id='next'>下一页 →</button></div></main><script src='assets/search-index.js'></script><script src='assets/review.js'></script></body></html>"""
    (review_root / "index.html").write_text(index_html, encoding="utf-8")
    manifest = {
        "version": SCREENING_VERSION, "generated_at": utc_now(), "unique_items": len(search_items),
        "page_size": REVIEW_PAGE_SIZE, "page_count": page_count, "category_counts": dict(sorted(category_counts.items())),
        "image_files": len(image_names), "source": str(cfg.source),
    }
    (review_root / "manifest.json").write_text(json_dumps(manifest, pretty=True) + "\n", encoding="utf-8")
    (review_root / "README.md").write_text(
        "# K12Vista paginated review\n\nOpen `index.html` directly or serve this directory with `python -m http.server`. "
        "The directory must be copied together because pages and images are external. Human annotations are stored in browser localStorage; use the export button regularly.\n",
        encoding="utf-8",
    )
    launcher = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>K12Vista人工复核</title><style>body{{font:16px/1.6 system-ui;max-width:760px;margin:60px auto;padding:20px}}a{{display:inline-block;background:#1769aa;color:white;padding:10px 16px;border-radius:8px;text-decoration:none}}</style></head><body><h1>K12Vista 分页人工复核工具</h1><p>旧版单文件画廊已替换。当前包含 {len(search_items)} 道唯一题目，共 {page_count} 页，每页最多 {REVIEW_PAGE_SIZE} 题，图片按需加载。</p><p><a href='review/index.html'>打开分页复核工具</a></p><p>请完整保留或下载 <code>review/</code> 目录，不能只复制本入口文件。</p></body></html>"""
    (cfg.output / "gallery.html").write_text(launcher, encoding="utf-8")


def finalize(cfg: Config, db: Checkpoint) -> Dict[str, Any]:
    # Recompute image-status counters from the authoritative quality checkpoint.
    # Counter omitted zero-valued fields in early audits, but explicit zeros are
    # important in the final report (for example, ``damaged_image: 0``).
    quality = db.quality_map()
    dataset_summary = dict(db.get_meta("dataset_summary", {}) or {})
    image_status = {
        "with_image": sum(bool(v.get("image", {}).get("present")) for v in quality.values()),
        "without_image": sum(not bool(v.get("image", {}).get("present")) for v in quality.values()),
        "decodable_image": sum(bool(v.get("image", {}).get("decodable")) for v in quality.values()),
        "damaged_image": sum(
            bool(v.get("image", {}).get("present")) and not bool(v.get("image", {}).get("decodable"))
            for v in quality.values()
        ),
        "blank_image_flag": sum(bool(v.get("image", {}).get("blank_flag")) for v in quality.values()),
    }
    dataset_summary["image_status"] = image_status
    missing = dict(dataset_summary.get("missing", {}) or {})
    for key in ("question", "image", "damaged_image", "blank_image_flag", "answer", "solution", "knowledge_points"):
        missing.setdefault(key, 0)
    dataset_summary["missing"] = dict(sorted(missing.items()))
    (cfg.output / "dataset_summary.json").write_text(json_dumps(dataset_summary, pretty=True) + "\n", encoding="utf-8")
    db.set_meta("dataset_summary", dataset_summary)
    export_stage_files(cfg, db)
    records = make_final_records(cfg, db)
    write_jsonl(cfg.output / "screening_results.jsonl", records)
    categories = {category: [r for r in records if r.get("final_category") == category] for category in FINAL_CATEGORIES}
    write_jsonl(cfg.output / "core_candidates.jsonl", categories["A_CORE"])
    write_jsonl(cfg.output / "extension_candidates.jsonl", categories["B_EXTENSION"])
    write_jsonl(cfg.output / "static_candidates.jsonl", categories["C_STATIC"])
    write_jsonl(cfg.output / "rejected_samples.jsonl", categories["D_REJECT"])
    manual = [r for r in records if r.get("manual_review_required") or r.get("final_category") is None]
    write_jsonl(cfg.output / "manual_review.jsonl", manual)
    # A compact first-pass queue: every unresolved API/schema case first, then
    # diverse low-confidence A/B candidates that most affect benchmark quality.
    manual_priority = sorted(
        [r for r in manual if r.get("final_category") is None],
        key=lambda r: stable_hash(str(r["id"])),
    )
    priority_seen = {r["id"] for r in manual_priority}
    subject_caps: Counter[str] = Counter()
    priority_candidates = sorted(
        [r for r in manual if r["id"] not in priority_seen],
        key=lambda r: (
            0 if r.get("final_category") in {"A_CORE", "B_EXTENSION"} else 1,
            0 if r.get("conflict_flag") else 1,
            float(r.get("confidence") or 0),
            stable_hash(str(r["id"])),
        ),
    )
    for row in priority_candidates:
        subject = str(row.get("subject") or "unknown")
        if subject_caps[subject] >= 2:
            continue
        manual_priority.append(row); subject_caps[subject] += 1
        if len(manual_priority) >= 30:
            break
    if len(manual_priority) < min(30, len(manual)):
        priority_seen = {r["id"] for r in manual_priority}
        manual_priority.extend([r for r in priority_candidates if r["id"] not in priority_seen][:30-len(manual_priority)])
    write_jsonl(cfg.output / "manual_review_priority.jsonl", manual_priority)
    csv_path = cfg.output / "screening_results.csv"
    headers = ["id", "source_file", "subject", "grade", "question_type", "knowledge_points", "prefilter_label",
               "prefilter_confidence", "final_category", "decision_source", "simulation_family", "recommended_backend",
               *SCORE_NAMES, "manual_review_required", "confidence", "selection_reason"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers); writer.writeheader()
        for record in records:
            row = dict(record); row["knowledge_points"] = " | ".join(record.get("knowledge_points") or [])
            row.update(record.get("scores") or {}); writer.writerow({key: row.get(key) for key in headers})
    # Diverse, high-confidence first-round A candidates.
    ranked = sorted(
        [r for r in categories["A_CORE"] if r.get("simulation_family") not in {None, "", "none"}
         and r.get("recommended_backend") not in {None, "", "none"}],
        key=lambda r: (-float(r.get("confidence") or 0), stable_hash(str(r["id"]))),
    )
    recommendations: List[Dict[str, Any]] = []; used = Counter()
    for row in ranked:
        key = (str(row.get("subject")), str(row.get("simulation_family")))
        if used[key] >= 2: continue
        recommendations.append(row); used[key] += 1
        if len(recommendations) >= 20: break
    if len(recommendations) < min(10, len(ranked)):
        present = {r["id"] for r in recommendations}
        recommendations.extend([r for r in ranked if r["id"] not in present][:20-len(recommendations)])
    write_jsonl(cfg.output / "first_round_recommendations.jsonl", recommendations)
    deep = db.stage_map("deepseek"); qwen = db.stage_map("qwen")
    # Stage rows keep the latest state for resumability. Historical failed runs
    # remain in the append-only errors table, so combine successful-run attempts
    # with error attempts to report actual calls across all resumptions.
    def historical_calls(stage_name: str, states: Mapping[str, Mapping[str, Any]]) -> Tuple[int, int, int]:
        successful_attempts = sum(v["attempts"] for v in states.values() if v["status"] == "success")
        error_runs, failed_attempts = db.connection().execute(
            "SELECT COUNT(*),COALESCE(SUM(attempt),0) FROM errors WHERE stage=?", (stage_name,)
        ).fetchone()
        return int(successful_attempts + failed_attempts), int(error_runs), int(failed_attempts)

    deep_calls, deep_error_runs, deep_failed_attempts = historical_calls("deepseek", deep)
    qwen_calls, qwen_error_runs, qwen_failed_attempts = historical_calls("qwen", qwen)
    deep_input = sum(v["input_tokens"] for v in deep.values())
    deep_output = sum(v["output_tokens"] for v in deep.values())
    qwen_input = sum(v["input_tokens"] for v in qwen.values())
    qwen_output = sum(v["output_tokens"] for v in qwen.values())
    # Pricing snapshot used for this completed run (2026-08-06): qwen3-vl-plus
    # Beijing <=32K is CNY 1/M input and CNY 10/M output; DeepSeek-V4-Flash is
    # USD 0.0028/M cache-hit or USD 0.14/M cache-miss input and USD 0.28/M output.
    qwen_estimated_cny = qwen_input / 1_000_000 + 10 * qwen_output / 1_000_000
    deep_output_usd = 0.28 * deep_output / 1_000_000
    deep_min_usd = 0.0028 * deep_input / 1_000_000 + deep_output_usd
    deep_max_usd = 0.14 * deep_input / 1_000_000 + deep_output_usd
    usage = {
        "deepseek_call_count": deep_calls,
        "deepseek_success_count": sum(v["status"] == "success" for v in deep.values()),
        "deepseek_retry_count": max(0, deep_calls - len(deep)),
        "deepseek_error_run_count": deep_error_runs,
        "deepseek_failed_attempt_count": deep_failed_attempts,
        "deepseek_input_tokens": deep_input,
        "deepseek_output_tokens": deep_output,
        "qwen_call_count": qwen_calls,
        "qwen_success_count": sum(v["status"] == "success" for v in qwen.values()),
        "qwen_retry_count": max(0, qwen_calls - len(qwen)),
        "qwen_error_run_count": qwen_error_runs,
        "qwen_failed_attempt_count": qwen_failed_attempts,
        "qwen_input_tokens": qwen_input,
        "qwen_output_tokens": qwen_output,
        "estimated_cost": {
            "qwen_cny": round(qwen_estimated_cny, 2),
            "deepseek_usd_min_all_input_cache_hit": round(deep_min_usd, 2),
            "deepseek_usd_max_all_input_cache_miss": round(deep_max_usd, 2),
        },
        "estimated_cost_note": "Token totals are recorded usage for the latest successful/final run per item; some failed schema responses may have billed tokens that the API client could not recover, so costs are estimates. Workspace discounts are not included.",
        "pricing_sources": {
            "qwen": "https://help.aliyun.com/zh/model-studio/qwen3-vl-plus",
            "deepseek": "https://api-docs.deepseek.com/quick_start/pricing/",
        },
    }
    (cfg.output / "api_usage.json").write_text(json_dumps(usage, pretty=True) + "\n", encoding="utf-8")
    cat_counts = Counter(str(r.get("final_category") or "MANUAL_REVIEW") for r in records)
    subject_selected = Counter(str(r.get("subject")) for r in records if r.get("final_category") in {"A_CORE", "B_EXTENSION"})
    unresolved = [r for r in records if r.get("final_category") is None]
    unresolved_without_manual = [r for r in unresolved if not r.get("manual_review_required")]
    qwen_violation = [r["id"] for r in records if r.get("final_category") in {"A_CORE", "B_EXTENSION"} and r.get("decision_source") != "qwen_multimodal"]
    jsonl_count = len(records)
    # CSV fields such as model reasons may legally contain embedded newlines, so
    # physical line counting overstates the number of exported records. Parse
    # the CSV and count logical rows instead.
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        csv_count = sum(1 for _ in csv.DictReader(handle))
    completion = {
        "total_quality_records": len(db.quality_map()), "screening_records": jsonl_count, "csv_records": csv_count,
        "categories": dict(sorted(cat_counts.items())), "selected_by_subject": dict(sorted(subject_selected.items())),
        "manual_review": len(manual), "unresolved": len(unresolved),
        "unresolved_without_manual": len(unresolved_without_manual), "a_b_without_qwen": qwen_violation,
        "deepseek_failed": sum(v["status"] != "success" for v in deep.values()),
        "qwen_failed": sum(v["status"] != "success" for v in qwen.values()),
        "manual_review_priority_ids": [r["id"] for r in manual_priority],
        "recommendation_ids": [r["id"] for r in recommendations],
        "complete": bool(jsonl_count == csv_count == len(db.quality_map()) and not qwen_violation and not unresolved_without_manual),
        "generated_at": utc_now(),
    }
    (cfg.output / "completion_check.json").write_text(json_dumps(completion, pretty=True) + "\n", encoding="utf-8")
    # Replace a stale credit/API pause marker from an earlier resumable run.
    (cfg.output / "blocked_status.json").write_text(json_dumps({
        "status": "complete" if completion["complete"] else "needs_review",
        "completion_criteria_satisfied": completion["complete"],
        "generated_at": completion["generated_at"],
    }, pretty=True) + "\n", encoding="utf-8")
    summary = [
        "# K12Vista executable-simulation screening", "", f"- Version: `{SCREENING_VERSION}`",
        f"- Records: {jsonl_count}", f"- Categories: `{json_dumps(dict(cat_counts))}`",
        f"- Manual review: {len(manual)}", f"- DeepSeek calls/success: {usage['deepseek_call_count']}/{usage['deepseek_success_count']}",
        f"- Qwen calls/success: {usage['qwen_call_count']}/{usage['qwen_success_count']}",
        f"- Completion criteria satisfied: **{completion['complete']}**", "",
        "Failed API or schema cases are routed to manual review and are never auto-rejected.",
    ]
    (cfg.output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    readme = """# K12Vista selected

This directory is generated by `run_k12_screening.py`. `checkpoint.sqlite3` is the resumable
authoritative state. JSONL/CSV files are deterministic exports. Original K12Vista data remains
read-only and API credentials are never stored. `gallery.html` is the lightweight entry point for
the paginated tool under `review/`; copy the entire `review/` directory when moving it locally.
"""
    (cfg.output / "README.md").write_text(readme, encoding="utf-8")
    generate_gallery(cfg, records)
    db.set_meta("completion_check", completion)
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen full K12Vista for executable simulation tasks")
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", default="data/K12-Vista/raw/K12_Vista.jsonl")
    parser.add_argument("--output", default="output/k12vista-selected")
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--phase", choices=("audit", "smoke", "screen", "finalize", "review", "all"), default="all")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--deepseek-workers", type=int, default=12)
    parser.add_argument("--qwen-workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args); cfg.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "review":
        result_path = cfg.output / "screening_results.jsonl"
        if not result_path.is_file(): raise RuntimeError(f"missing completed screening results: {result_path}")
        with result_path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        generate_gallery(cfg, records)
        manifest = json.loads((cfg.output / "review" / "manifest.json").read_text(encoding="utf-8"))
        print(json_dumps(manifest, pretty=True), flush=True)
        return 0
    db = Checkpoint(cfg.output / "checkpoint.sqlite3")
    db.set_meta("run_config", {
        "screening_version": SCREENING_VERSION, "text_prompt_version": TEXT_PROMPT_VERSION,
        "vision_prompt_version": VISION_PROMPT_VERSION, "deepseek_model": cfg.deepseek_model,
        "qwen_model": cfg.qwen_model, "seed": cfg.seed, "source": str(cfg.source),
        "started_or_resumed_at": utc_now(),
    })
    if args.phase in {"audit", "all"}: audit_dataset(cfg, db)
    if args.phase in {"smoke", "all"}:
        if not db.get_meta("audit_complete", False): audit_dataset(cfg, db)
        smoke_test(cfg, db)
    if args.phase in {"screen", "all"}:
        if not db.get_meta("smoke_complete"): smoke_test(cfg, db)
        full_screen(cfg, db)
    if args.phase in {"finalize", "all"}:
        completion = finalize(cfg, db); print(json_dumps(completion, pretty=True), flush=True)
        return 0 if completion.get("complete") else 2
    return 0
