"""Build a small diagnostic subset from the first 10 LongMemEval-S questions.

Keep every labeled evidence session and the two nearest non-evidence sessions.
This uses answer-location labels to reduce the haystack and therefore must not
be reported as an official LongMemEval-S score.
"""

from __future__ import annotations

import json
from typing import Any

from scripts.run_longmemeval import DEFAULT_DATA, PROJECT_ROOT

OUTPUT = (
    PROJECT_ROOT
    / "data/memory_eval/longmemeval/datasets/first10_evidence_plus_2distractors.json"
)
QUESTION_COUNT = 10
DISTRACTOR_COUNT = 2


def slim_case(entry: dict[str, Any]) -> dict[str, Any]:
    session_ids = entry["haystack_session_ids"]
    dates = entry["haystack_dates"]
    sessions = entry["haystack_sessions"]
    if not (len(session_ids) == len(dates) == len(sessions)):
        raise ValueError(f"{entry['question_id']}: 会话字段长度不一致")
    evidence_ids = set(entry["answer_session_ids"])
    if not evidence_ids:
        raise ValueError(f"{entry['question_id']}: 没有证据会话，不适用这个精简规则")
    evidence_indexes = {i for i, session_id in enumerate(session_ids) if session_id in evidence_ids}
    if len(evidence_indexes) != len(evidence_ids):
        raise ValueError(f"{entry['question_id']}: 部分证据会话不在历史中")

    candidates = [i for i in range(len(session_ids)) if i not in evidence_indexes]
    # Prefer nearby sessions; ties retain the earlier one. Keep original order below.
    candidates.sort(key=lambda i: (min(abs(i - j) for j in evidence_indexes), i))
    selected_ids = {session_ids[i] for i in evidence_indexes}
    distractor_indexes = []
    for index in candidates:
        # Some official histories reuse a non-evidence session ID. The replay
        # storage keys by session ID, so do not select colliding sessions.
        if session_ids[index] in selected_ids:
            continue
        distractor_indexes.append(index)
        selected_ids.add(session_ids[index])
        if len(distractor_indexes) == DISTRACTOR_COUNT:
            break
    if len(distractor_indexes) != DISTRACTOR_COUNT:
        raise ValueError(f"{entry['question_id']}: 可用的不同 ID 干扰会话不足")
    selected = sorted(evidence_indexes | set(distractor_indexes))

    return {
        **entry,
        "haystack_session_ids": [session_ids[i] for i in selected],
        "haystack_dates": [dates[i] for i in selected],
        "haystack_sessions": [sessions[i] for i in selected],
    }


def main() -> None:
    source = json.loads(DEFAULT_DATA.read_text(encoding="utf-8"))
    if not isinstance(source, list) or len(source) < QUESTION_COUNT:
        raise ValueError("LongMemEval 数据不足 10 题")
    subset = [slim_case(entry) for entry in source[:QUESTION_COUNT]]
    payload = json.dumps(subset, ensure_ascii=False, indent=2) + "\n"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        if OUTPUT.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"文件已存在且内容不同，不会覆盖：{OUTPUT}")
        print(f"精简数据已存在且内容一致：{OUTPUT}")
    else:
        OUTPUT.write_text(payload, encoding="utf-8")
        print(f"已生成精简数据：{OUTPUT}")

    for original, slimmed in zip(source[:QUESTION_COUNT], subset, strict=True):
        before = sum(len(session) for session in original["haystack_sessions"])
        after = sum(len(session) for session in slimmed["haystack_sessions"])
        print(
            f"{original['question_id']} [{original['question_type']}]: "
            f"{len(original['haystack_sessions'])}→{len(slimmed['haystack_sessions'])} 段，"
            f"{before}→{after} 条消息"
        )


if __name__ == "__main__":
    main()
