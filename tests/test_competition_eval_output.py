from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_competition_eval_module():
    spec = importlib.util.spec_from_file_location(
        "competition_eval_under_test",
        ROOT / "scripts" / "competition_eval.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_csv_contains_questions(tmp_path):
    module = load_competition_eval_module()
    output_path = tmp_path / "submission.csv"

    module._write_submission_csv(
        [
            {
                "id": 317,
                "question": "When using the grill, how to connect regulator to the LP Tank?",
                "ret": "Answer text",
            }
        ],
        output_path,
    )

    with output_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {
            "id": "317",
            "question": "When using the grill, how to connect regulator to the LP Tank?",
            "ret": "Answer text",
        }
    ]
