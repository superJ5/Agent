"""Patch extract_query_terms to use TF-IDF based extraction."""
import pathlib

target = pathlib.Path(r"e:\.codex\worktrees\3f3c\ai_agent_competition\app\tools\knowledge_tool.py")
content = target.read_text(encoding="utf-8")

# Find the function
marker = "def extract_query_terms(query: str) -> list[str]:"
start = content.find(marker)
if start < 0:
    raise RuntimeError("Cannot find extract_query_terms function")

# Find end: next top-level def/class after it
rest = content[start:]
lines = rest.split("\n")
end_idx = len(lines)
for i, line in enumerate(lines):
    if i > 0 and line and not line[0].isspace() and not line.startswith("#"):
        end_idx = i
        break

# Trim trailing blank lines
while end_idx > 0 and lines[end_idx - 1].strip() == "":
    end_idx -= 1

old_block = "\n".join(lines[:end_idx])

new_block = '''def extract_query_terms(query: str) -> list[str]:
    """Extract high-signal terms from the query using TF-IDF segmentation."""
    keywords: list[str] = []

    # 1. TF-IDF 提取关键词（不依赖任何词表）
    try:
        from importlib import import_module
        jieba_analyse = import_module("jieba.analyse")
        keywords = list(jieba_analyse.extract_tags(query, topK=8, withWeight=False))
    except Exception:
        # jieba 不可用时 fallback 到正则粗分词
        keywords.extend(re.findall(r"[\\u4e00-\\u9fff]{2,}", query))
        keywords.extend(re.findall(r"[A-Za-z0-9_\\-]{3,}", query))

    # 2. 补充图片 ID
    pic_id = extract_pic_id(query)
    if pic_id:
        keywords.insert(0, pic_id)

    # 3. 补充 profile 术语（领域加权）
    normalized = normalize_text(query)
    keywords.extend(match_profile_terms_in_query(normalized, limit=4))

    # 4. 去重
    if keywords:
        return list(dict.fromkeys(keywords))[:16]

    # 5. 兜底
    compact = re.sub(r"\\s+", "", query)
    return [compact[:12]] if compact else []'''

content = content.replace(old_block, new_block, 1)
target.write_text(content, encoding="utf-8")
print("SUCCESS: extract_query_terms has been patched.")
