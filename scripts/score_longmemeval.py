"""Score existing LongMemEval answers with the configured DashScope judge.

This follows LongMemEval's QA judging prompts, but a Qwen-judged score is not
the official GPT-4o-judged benchmark score. No history is replayed here.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from openai import OpenAI

from app.config import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES = (
    PROJECT_ROOT / "data/memory_eval/longmemeval/datasets/first10_evidence_plus_2distractors.json"
)


def judge_prompt(question_type: str, question: str, answer: str, hypothesis: str, *, abstention: bool) -> str:
    if abstention:
        instruction = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is "
            "given but the asked information is not."
        )
        answer_label = "Explanation"
        closing = "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    elif question_type == "single-session-preference":
        instruction = (
            "I will give you a question, a rubric for desired personalized response, and a response "
            "from a model. Please answer yes if the response satisfies the desired response. Otherwise, "
            "answer no. The model does not need to reflect all the points in the rubric. The response "
            "is correct as long as it recalls and utilizes the user's personal information correctly."
        )
        answer_label = "Rubric"
        closing = "Is the model response correct? Answer yes or no only."
    elif question_type == "knowledge-update":
        instruction = (
            "I will give you a question, a correct answer, and a response from a model. Please answer "
            "yes if the response contains the correct answer. Otherwise, answer no. If the response "
            "contains some previous information along with an updated answer, the response should be "
            "considered as correct as long as the updated answer is the required answer."
        )
        answer_label = "Correct Answer"
        closing = "Is the model response correct? Answer yes or no only."
    elif question_type in {"single-session-user", "single-session-assistant", "multi-session", "temporal-reasoning"}:
        instruction = (
            "I will give you a question, a correct answer, and a response from a model. Please answer "
            "yes if the response contains the correct answer. Otherwise, answer no. If the response is "
            "equivalent to the correct answer or contains all the intermediate steps to get the correct "
            "answer, you should also answer yes. If the response only contains a subset of the "
            "information required by the answer, answer no."
        )
        if question_type == "temporal-reasoning":
            instruction += (
                " In addition, do not penalize off-by-one errors for the number of days. If the "
                "question asks for the number of days/weeks/months, etc., and the model makes off-by-one "
                "errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct."
            )
        answer_label = "Correct Answer"
        closing = "Is the model response correct? Answer yes or no only."
    else:
        raise ValueError(f"不支持的题目类型: {question_type}")
    return (
        f"{instruction}\n\nQuestion: {question}\n\n{answer_label}: {answer}"
        f"\n\nModel Response: {hypothesis}\n\n{closing}"
    )


def load_hypotheses(paths: list[Path]) -> list[dict]:
    rows = [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["question_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("答案文件包含重复的 question_id；请仅传入每题最终采用的一次运行结果")
    return rows


def parse_label(raw: str) -> bool:
    label = raw.strip().lower()
    if not re.fullmatch(r"(?:yes|no)[.!。]?", label):
        raise ValueError(f"裁判未按要求输出 yes/no: {raw!r}")
    return label.startswith("yes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hypotheses", nargs="+", type=Path, help="一个或多个回放结果 JSONL")
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--output", type=Path, required=True, help="逐题评分结果 JSONL；拒绝覆盖已有文件")
    parser.add_argument("--model", default=config.rag_model)
    args = parser.parse_args()

    hypotheses = load_hypotheses(args.hypotheses)
    references = {str(row["question_id"]): row for row in json.loads(args.references.read_text(encoding="utf-8"))}
    missing = [row["question_id"] for row in hypotheses if str(row["question_id"]) not in references]
    if missing:
        parser.error(f"标准答案中不存在这些题: {missing}")
    if not hypotheses:
        parser.error("答案文件为空")
    if not config.dashscope_api_key:
        parser.error(".env 中缺少 DASHSCOPE_API_KEY")
    if args.output.exists():
        parser.error(f"评分文件已存在，不覆盖: {args.output}")

    client = OpenAI(api_key=config.dashscope_api_key, base_url=config.dashscope_api_base, timeout=60.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with args.output.open("x", encoding="utf-8") as out:
        for index, row in enumerate(hypotheses, 1):
            qid = str(row["question_id"])
            ref = references[qid]
            prompt = judge_prompt(
                ref["question_type"], ref["question"], ref["answer"], str(row["hypothesis"]),
                abstention="_abs" in qid,
            )
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=32,
                extra_body={"enable_thinking": False},
            )
            raw = response.choices[0].message.content or ""
            label = parse_label(raw)
            correct += label
            result = {
                "question_id": qid,
                "question_type": ref["question_type"],
                "answer": ref["answer"],
                "hypothesis": row["hypothesis"],
                "judge_model": args.model,
                "judge_response": raw,
                "correct": label,
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{index}/{len(hypotheses)}] {qid}: {'正确' if label else '错误'}", flush=True)
    print(f"Qwen 裁判准确率: {correct}/{len(hypotheses)} = {correct / len(hypotheses):.1%}")
    print(f"逐题结果: {args.output}")


if __name__ == "__main__":
    main()
