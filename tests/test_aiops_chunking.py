from app.services.aiops_chunking import chunk_markdown_document


def test_chunk_markdown_keeps_heading_hierarchy_and_skips_empty_parents():
    content = """# CPU 告警

## 排查步骤

### 查询指标

查询最近一小时的 CPU 使用率。
"""

    records = chunk_markdown_document(content, "aiops-docs/cpu.md")

    assert len(records) == 1
    assert records[0]["section_path"] == ["CPU 告警", "排查步骤", "查询指标"]
    assert "文档：CPU 告警" in records[0]["text"]
    assert "查询最近一小时的 CPU 使用率" in records[0]["text"]


def test_chunk_markdown_does_not_treat_shell_comment_in_fence_as_heading():
    content = """# 磁盘告警

## 常用命令

```bash
# 查看磁盘使用率
df -h
```
"""

    records = chunk_markdown_document(content, "aiops-docs/disk.md")

    assert len(records) == 1
    assert records[0]["section_path"] == ["磁盘告警", "常用命令"]
    assert "# 查看磁盘使用率" in records[0]["text"]


def test_chunk_markdown_splits_oversized_section():
    content = "# 内存告警\n\n## 原因分析\n\n" + ("内存泄漏现象。" * 100)

    records = chunk_markdown_document(content, "aiops-docs/memory.md", max_chars=240)

    assert len(records) > 1
    assert all(len(record["text"]) <= 240 for record in records)
