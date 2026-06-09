"""Local routing rules for manual RAG and generic customer-service queries."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANUAL_CHUNK_DIR = PROJECT_ROOT / "data" / "manuals" / "chunks"

PIC_ID_RE = re.compile(r"\bmanual\d+_[a-z0-9_]+\b", re.IGNORECASE)
MODEL_ID_RE = re.compile(r"\b(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z][a-z0-9-]{3,}\b", re.IGNORECASE)
ENGLISH_WORD_RE_TEMPLATE = r"(?<![a-z0-9]){}(?![a-z0-9])"

GENERIC_CUSTOMER_TERMS = (
    "商品",
    "订单",
    "下单",
    "付款",
    "支付",
    "退款",
    "退货",
    "换货",
    "退换货",
    "售后",
    "客服",
    "物流",
    "快递",
    "运费",
    "配送",
    "发票",
    "优惠券",
    "投诉",
    "赔偿",
    "补发",
    "补寄",
    "少发",
    "漏发",
    "包装破损",
    "质量问题",
    "详情页",
    "上门安装服务",
    "试用装",
    "七天无理由",
    "7天无理由",
    "customer service",
    "order",
    "shipping",
    "delivery",
    "invoice",
    "refund",
    "return",
    "coupon",
)

GENERIC_CUSTOMER_CONTEXT_TERMS = (
    "我购买",
    "我收到",
    "你们家",
    "你们的商品",
    "申请售后",
    "联系售后",
    "联系平台",
    "上门人员",
    "安装人员",
    "检修人员",
    "维修人员",
    "额外收取",
    "收取费用",
    "承诺的",
    "该怎么处理",
)

EXPLICIT_MANUAL_REFERENCE_TERMS = (
    "根据手册",
    "按照手册",
    "手册中",
    "说明书中",
    "根据说明书",
    "原文",
    "章节",
    "图示",
    "ocr",
    "according to the manual",
    "in the manual",
    "manual program",
)

MANUAL_INTENT_TERMS = (
    "如何",
    "怎么",
    "怎样",
    "是什么",
    "有哪些",
    "指什么",
    "作用",
    "功能",
    "步骤",
    "安装",
    "装入",
    "设置",
    "连接",
    "配对",
    "操作",
    "使用",
    "启动",
    "关闭",
    "调节",
    "拆卸",
    "更换",
    "清洁",
    "组装",
    "校准",
    "复位",
    "重置",
    "指示灯",
    "按钮",
    "按键",
    "接口",
    "部件",
    "参数",
    "规格",
    "故障",
    "排查",
    "安全",
    "警告",
    "how to",
    "how the",
    "how do",
    "how can",
    "how should",
    "what is",
    "what are",
    "what do",
    "what can",
    "what should",
    "what kinds",
    "what information",
    "please introduce",
    "if i need to",
    "when encountering",
    "difficult to",
    "first use",
    "install",
    "setup",
    "connect",
    "pair",
    "operate",
    "replace",
    "clean",
    "assemble",
    "button",
    "indicator",
    "component",
    "specification",
    "troubleshoot",
    "safety",
)


def should_use_manual_rag(question: str, *, has_images: bool = False) -> bool:
    """Return whether a query should be allowed to retrieve product manuals."""
    normalized = _normalize(question)
    if not normalized:
        return False

    if has_images or PIC_ID_RE.search(normalized):
        return True

    if _contains_product_name(normalized) or MODEL_ID_RE.search(normalized):
        return True

    if any(term in normalized for term in EXPLICIT_MANUAL_REFERENCE_TERMS):
        return True

    if any(term in normalized for term in GENERIC_CUSTOMER_TERMS):
        return False

    if any(term in normalized for term in GENERIC_CUSTOMER_CONTEXT_TERMS):
        return False

    return any(term in normalized for term in MANUAL_INTENT_TERMS)


@lru_cache(maxsize=1)
def load_product_names() -> tuple[str, ...]:
    """Derive product names from the local manual chunk filenames."""
    names: set[str] = set()
    if not MANUAL_CHUNK_DIR.exists():
        return ()

    for path in MANUAL_CHUNK_DIR.glob("*.jsonl"):
        parts = path.stem.split("__")
        if len(parts) < 2:
            continue
        raw_name = parts[1].strip()
        if not raw_name:
            continue
        names.add(_normalize(raw_name))
        without_manual = re.sub(r"手册$", "", raw_name, flags=re.IGNORECASE).strip()
        if without_manual:
            names.add(_normalize(without_manual))

    return tuple(sorted((name for name in names if len(name) >= 2), key=len, reverse=True))


def _contains_product_name(normalized_question: str) -> bool:
    for product_name in load_product_names():
        if re.fullmatch(r"[a-z0-9 _-]+", product_name):
            pattern = ENGLISH_WORD_RE_TEMPLATE.format(re.escape(product_name))
            if re.search(pattern, normalized_question):
                return True
        elif product_name in normalized_question:
            return True
    return False


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())
