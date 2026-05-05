"""End-to-End Test for RAG Pipeline: Query -> Retrieval -> LLM Answer

This script allows you to test the complete pipeline:
1. User enters a query
2. The agent uses `retrieve_knowledge` to query Milvus
3. The LLM processes the retrieved chunks and generates an answer

Usage:
    python scripts/test_e2e_rag.py "键盘的USB接口在哪？"
"""

import argparse
import asyncio
import sys
import uuid

# Patch the path to ensure we can import 'app'
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.rag_agent_service import rag_agent_service

async def run_e2e_test(query: str):
    session_id = str(uuid.uuid4())
    print(f"\n{'='*60}")
    print(f"[USER QUERY]: {query}")
    print(f"{'='*60}")

    print("Agent is thinking and retrieving...\n")
    try:
        # Call the non-streaming method to get the final answer
        # The agent will internally call the tool 'retrieve_knowledge'
        answer = await rag_agent_service.query(query, session_id=session_id)
        
        print(f"\n{'='*60}")
        print("[LLM ANSWER]:")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n[ERROR]: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run E2E test of the RAG pipeline.")
    parser.add_argument("query", type=str, help="The question to ask the RAG agent.")
    args = parser.parse_args()

    asyncio.run(run_e2e_test(args.query))

if __name__ == "__main__":
    main()
