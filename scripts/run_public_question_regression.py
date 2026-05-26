"""Run batch public-question chat regression cases against the RAG agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class QueryCase:
    case_id: str
    query: str
    doc_id: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run public-question chat regression cases through rag_agent_service.",
    )
    parser.add_argument(
        "--queries-file",
        required=True,
        type=Path,
        help="Input file containing questions in JSONL, JSON, or TXT format.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("reports/public_question_runs/chat"),
        type=Path,
        help="Directory for timestamped JSON and Markdown reports.",
    )
    parser.add_argument(
        "--case-prefix",
        default="chat",
        help="Prefix used when a case_id must be generated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases to run.",
    )

    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be greater than or equal to 0")
    return args


def load_cases(queries_file: Path, case_prefix: str, limit: int | None) -> list[QueryCase]:
    path = queries_file if queries_file.is_absolute() else Path.cwd() / queries_file
    if not path.exists():
        raise FileNotFoundError(f"Queries file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        raw_cases = _load_jsonl(path)
    elif suffix == ".json":
        raw_cases = _load_json(path)
    else:
        raw_cases = _load_txt(path)

    cases: list[QueryCase] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if limit is not None and len(cases) >= limit:
            break
        cases.append(_normalize_case(raw_case, index, case_prefix))
    return cases


def _load_jsonl(path: Path) -> list[Any]:
    cases: list[Any] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                cases.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return cases


def _load_json(path: Path) -> list[Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {exc}") from exc

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("cases", "queries", "questions", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        if "query" in data:
            return [data]

    raise ValueError(
        "JSON input must be an object with a query field, a list of objects, "
        "or an object containing a cases/queries/questions/items list."
    )


def _load_txt(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_case(raw_case: Any, index: int, case_prefix: str) -> QueryCase:
    fallback_case_id = f"{case_prefix}_{index:03d}"

    if isinstance(raw_case, str):
        query = raw_case.strip()
        if not query:
            raise ValueError(f"Case {index} has an empty query")
        return QueryCase(case_id=fallback_case_id, query=query)

    if not isinstance(raw_case, dict):
        raise ValueError(f"Case {index} must be an object or text line")

    raw_query = raw_case.get("query")
    if not isinstance(raw_query, str) or not raw_query.strip():
        raise ValueError(f"Case {index} must contain a non-empty string query")

    raw_case_id = raw_case.get("case_id")
    case_id = str(raw_case_id).strip() if raw_case_id is not None else fallback_case_id
    if not case_id:
        case_id = fallback_case_id

    raw_doc_id = raw_case.get("doc_id")
    doc_id = str(raw_doc_id).strip() if raw_doc_id is not None else None
    if doc_id == "":
        doc_id = None

    return QueryCase(case_id=case_id, query=raw_query.strip(), doc_id=doc_id)


async def run_cases(cases: list[QueryCase]) -> list[dict[str, Any]]:
    try:
        _initialize_retrieval_dependencies()
        from app.services.rag_agent_service import rag_agent_service
    except Exception as exc:
        return [_build_blocked_case_result(case, exc) for case in cases]

    results: list[dict[str, Any]] = []
    for case in cases:
        results.append(await run_case(case, rag_agent_service))
    return results


def _initialize_retrieval_dependencies() -> None:
    """Mirror FastAPI startup for scripts that bypass app.main lifespan."""
    from app.core.milvus_client import milvus_manager
    from app.retrieval.bm25_provider import init_bm25_provider

    milvus_manager.connect()
    init_bm25_provider()


def _build_blocked_case_result(case: QueryCase, exc: Exception) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_id": case.case_id,
        "query": case.query,
        "session_id": str(uuid.uuid4()),
        "answer": None,
        "metadata": None,
        "duration_ms": 0,
        "status": "blocked",
        "error": f"{type(exc).__name__}: {exc}",
    }
    if case.doc_id is not None:
        result["doc_id"] = case.doc_id
    return result


async def run_case(case: QueryCase, rag_agent_service: Any) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    started_at = time.perf_counter()
    answer: str | None = None
    error: str | None = None
    status = "succeeded"

    try:
        answer = await rag_agent_service.query(case.query, session_id=session_id)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

    metadata = _get_metadata_safely(rag_agent_service, session_id)
    duration_ms = round((time.perf_counter() - started_at) * 1000)

    result: dict[str, Any] = {
        "case_id": case.case_id,
        "query": case.query,
        "session_id": session_id,
        "answer": answer,
        "metadata": metadata,
        "duration_ms": duration_ms,
        "status": status,
        "error": error,
    }
    if case.doc_id is not None:
        result["doc_id"] = case.doc_id
    return result


def _get_metadata_safely(rag_agent_service: Any, session_id: str) -> dict[str, Any] | None:
    try:
        metadata = rag_agent_service.get_last_retrieval_metadata(session_id)
    except Exception:
        return None
    return metadata if isinstance(metadata, dict) else None


def build_report(results: list[dict[str, Any]], json_path: Path, markdown_path: Path) -> dict[str, Any]:
    succeeded = sum(1 for case in results if case.get("status") == "succeeded")
    blocked = sum(1 for case in results if case.get("status") == "blocked")
    failed = len(results) - succeeded
    summary = {
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "blocked": blocked,
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": summary,
        "cases": results,
    }


def write_outputs(results: list[dict[str, Any]], output_dir: Path, case_prefix: str) -> tuple[Path, Path, dict[str, Any]]:
    resolved_output_dir = _resolve_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_prefix = _safe_filename_part(case_prefix) or "chat"
    json_path = resolved_output_dir / f"{safe_prefix}_{timestamp}.json"
    markdown_path = resolved_output_dir / f"{safe_prefix}_{timestamp}.md"

    report = build_report(results, json_path, markdown_path)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(format_markdown(report), encoding="utf-8")
    return json_path, markdown_path, report["summary"]


def _resolve_output_dir(output_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir
    return PROJECT_ROOT / output_dir


def _safe_filename_part(value: str) -> str:
    safe_chars = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_")


def format_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Public Question Chat Regression",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Total: {summary['total']}",
        f"- Succeeded: {summary['succeeded']}",
        f"- Failed: {summary['failed']}",
        "",
    ]

    for case in report["cases"]:
        lines.extend(_format_case_markdown(case))

    return "\n".join(lines).rstrip() + "\n"


def _format_case_markdown(case: dict[str, Any]) -> list[str]:
    lines = [
        f"## {case['case_id']} - {case['status']}",
        "",
        f"- Session: `{case['session_id']}`",
        f"- Duration: {case['duration_ms']} ms",
    ]
    if "doc_id" in case:
        lines.append(f"- Doc ID: `{case['doc_id']}`")

    lines.extend(
        [
            "",
            "### Query",
            "",
            _fenced_text(case.get("query") or ""),
            "",
            "### Answer",
            "",
            _fenced_text(case.get("answer") or ""),
        ]
    )

    if case.get("error"):
        lines.extend(["", "### Error", "", _fenced_text(case["error"])])

    lines.extend(
        [
            "",
            "### Metadata",
            "",
            _fenced_text(json.dumps(case.get("metadata"), ensure_ascii=False, indent=2, default=str), "json"),
            "",
        ]
    )
    return lines


def _fenced_text(value: str, language: str = "") -> str:
    fence = "```"
    while fence in value:
        fence += "`"
    suffix = language if language else ""
    return f"{fence}{suffix}\n{value}\n{fence}"


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.queries_file, args.case_prefix, args.limit)
    results = await run_cases(cases)
    json_path, markdown_path, summary = write_outputs(results, args.output_dir, args.case_prefix)
    return {
        "markdown": str(markdown_path),
        "json": str(json_path),
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        output = asyncio.run(async_main(args))
    except Exception as exc:
        print(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
