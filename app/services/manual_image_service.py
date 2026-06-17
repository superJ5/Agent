"""Resolve manual picture IDs to local image files for the web UI."""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

PIC_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANUAL_RAW_ROOT = PROJECT_ROOT / "data" / "manuals" / "raw"


class ManualImageNotFoundError(FileNotFoundError):
    """Raised when a manual image id cannot be resolved."""


def resolve_manual_image_path(image_id: str) -> Path:
    """Return a local image path for a picture id from images_manifest.csv files."""
    normalized_id = _normalize_image_id(image_id)
    path = _image_manifest_index().get(normalized_id)
    if path is None:
        path = _scan_image_by_stem(normalized_id)
    if path is None or not path.is_file():
        raise ManualImageNotFoundError(f"manual image not found: {normalized_id}")
    return path


def _normalize_image_id(image_id: str) -> str:
    normalized = str(image_id or "").strip()
    if not PIC_ID_RE.fullmatch(normalized):
        raise ManualImageNotFoundError(f"invalid manual image id: {image_id}")
    return normalized


@lru_cache(maxsize=1)
def _image_manifest_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not MANUAL_RAW_ROOT.exists():
        return index

    for manifest_path in MANUAL_RAW_ROOT.glob("*/images_manifest.csv"):
        manual_dir = manifest_path.parent
        try:
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    image_id = str(row.get("image_id") or "").strip()
                    relative_path = str(row.get("relative_path") or "").strip()
                    filename = str(row.get("filename") or "").strip()
                    if not image_id:
                        continue
                    candidate = manual_dir / (relative_path or f"images/{filename}")
                    resolved = _safe_resolve_under(candidate, MANUAL_RAW_ROOT)
                    if resolved is not None and resolved.is_file():
                        index.setdefault(image_id, resolved)
        except OSError:
            continue
    return index


def _scan_image_by_stem(image_id: str) -> Path | None:
    for suffix in ("jpg", "jpeg", "png", "webp"):
        for candidate in MANUAL_RAW_ROOT.glob(f"*/images/{image_id}.{suffix}"):
            resolved = _safe_resolve_under(candidate, MANUAL_RAW_ROOT)
            if resolved is not None and resolved.is_file():
                return resolved
    return None


def _safe_resolve_under(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None
