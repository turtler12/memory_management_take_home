# src/run_graph_demo.py
"""
Demo for GraphMemory with embeddings and hybrid recall.

Run:
  export NEO4J_URI=bolt://localhost:7687
  export NEO4J_USER=neo4j
  export NEO4J_PASSWORD=<password>
  # optional:
  # export OPENAI_API_KEY=...
  python -m src.run_graph_demo
"""

import os, random, json
from .graph_memory import GraphMemory

def fake_tool_output(tool, **kwargs):
    return {
        "tool": tool,
        "inputs": kwargs,
        "result": {"status": "OK", "data": {"value": random.randint(100, 999)}}
    }

def main():
    # small budget to show compression quickly
    os.environ.setdefault("MEM_TOKEN_BUDGET", "2000")

    mem = GraphMemory()
    mem.init_schema()

    prev = None
    ids = []

    # 1) call: aws.sts.get_caller_identity
    out1 = fake_tool_output("aws.sts.get_caller_identity")
    id1, hit = mem.remember_or_get(
        "aws.sts.get_caller_identity",
        inputs={},
        output=out1,
        prev_id=prev
    )
    print(f"[1] id={id1} hit={hit}")
    prev = id1; ids.append(id1)

    # 2) call: execute_command ls (derived from 1)
    out2 = fake_tool_output("execute_command", command="aws s3 ls")
    id2, hit = mem.remember_or_get(
        "execute_command",
        inputs={"command": "aws s3 ls"},
        output=out2,
        prev_id=prev,
        derived_from_ids=[id1]
    )
    print(f"[2] id={id2} hit={hit}")
    prev = id2; ids.append(id2)

    # 3) duplicate of step 2 (should be a HIT)
    out2b = fake_tool_output("execute_command", command="aws s3 ls")
    id2b, hit = mem.remember_or_get(
        "execute_command",
        inputs={"command": "aws s3 ls"},
        output=out2b,
        prev_id=prev
    )
    print(f"[2-dup] id={id2b} hit={hit}")

    # 4) modify_code (derived from 2)
    out3 = fake_tool_output("modify_code", files=["src/app.py"], instructions="add logging")
    id3, hit = mem.remember_or_get(
        "modify_code",
        inputs={"files": ["src/app.py"], "instructions": "add logging"},
        output=out3,
        prev_id=prev,
        derived_from_ids=[id2]
    )
    print(f"[3] id={id3} hit={hit}")
    ids.append(id3)

    # 5) Top salient
    print("\nTop salient:")
    for item in mem.top_salient(limit=5):
        print(item)

    # 6) Manual compression
    mem.compress_nodes([id1])
    print(f"\nCompressed node {id1}")

    # 7) Expand full body of id2
    full = mem.expand(id2)
    print("\nExpanded id2 body:", json.dumps(full)[:120], "...")

    # 8) Read node 3 props
    print("\nNode 3 props:", mem.get(id3))

    # 9) Ensure any missing embeddings are backfilled (usually not needed)
    mem.backfill_missing_embeddings()

    # 10) Hybrid recall around id3 (structural pre-filter + semantic re-rank)
    print("\nHybrid recall (around id3):")
    for item in mem.hybrid_recall(anchor_id=id3, hops=3, k=5):
        print(item)

    mem.close()

if __name__ == "__main__":
    main()
