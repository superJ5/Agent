path = r"e:\.codex\worktrees\3f3c\ai_agent_competition\app\services\vector_search_service.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

content2 = content.replace('"metric_type": "L2"', '"metric_type": "COSINE"', 1)
content2 = content2.replace('"nprobe": 10', '"ef": 64', 1)

if content2 != content:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content2)
    print("SUCCESS: metric_type updated to COSINE, nprobe -> ef")
else:
    print("NOT CHANGED - pattern not found")
