"""AIOps runbook retrieval tool."""

from langchain_core.tools import tool

from app.services.aiops_knowledge_service import aiops_knowledge_service


@tool
def retrieve_aiops_knowledge(query: str) -> str:
    """检索运维手册、故障案例和排查步骤，为 AIOps 诊断提供经验依据。

    Args:
        query: 需要查询的告警现象、服务故障或运维问题。
    """
    matches = aiops_knowledge_service.search(query)
    if not matches:
        return "未检索到相关运维知识。"

    sections: list[str] = []
    for index, match in enumerate(matches, 1):
        metadata = match["metadata"]
        sections.append(
            f"【运维知识 {index}】{metadata.get('doc_name', '')} / "
            f"{metadata.get('title', '')}\n"
            f"相似度：{match['score']:.4f}\n"
            f"{match['content']}"
        )
    return "\n\n".join(sections)
