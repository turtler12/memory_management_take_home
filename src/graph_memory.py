# src/graph_memory.py
"""
Graph-native memory for agent tool calls (Neo4j + optional OpenAI summaries + embeddings + decay/compress).

Env:
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
Optional:
  OPENAI_API_KEY, EMBED_MODEL (text-embedding-3-small), MEM_TOKEN_BUDGET

New:
  - t.salience property (float)
  - decay_and_compress(...) to recompute salience with temporal decay and auto-compress below cutoff
"""

from __future__ import annotations
import os, json, time, hashlib, math
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase

# ---- optional OpenAI (best-effort) ----
_USE_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
_EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
_openai_client = None
if _USE_OPENAI:
    try:
        from openai import OpenAI
        _openai_client = OpenAI()
    except Exception:
        _USE_OPENAI = False

# ---- config ----
_MEM_TOKEN_BUDGET = int(os.environ.get("MEM_TOKEN_BUDGET", "30000"))
_NEO4J_URI = os.environ.get("NEO4J_URI")
_NEO4J_USER = os.environ.get("NEO4J_USER")
_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")


def _now() -> float:
    return time.time()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _stable_key(tool_name: str, inputs: Dict[str, Any]) -> str:
    s = f"{tool_name}:{_canonical_json(inputs)}"
    return hashlib.sha256(s.encode()).hexdigest()


def _rough_token_estimate(s: str) -> int:
    return max(1, len(s) // 4)


def _one_line_summary(tool_name: str, inputs: dict, output: dict) -> str:
    head = f"{tool_name}(" + ", ".join(f"{k}={json.dumps(v)[:60]}" for k, v in sorted(inputs.items())) + ")"
    if isinstance(output, dict):
        keys = ",".join(list(output.keys())[:6])
        tail = f" -> keys:{keys}"
    else:
        tail = f" -> {str(output)[:80]}"
    return (head + tail)[:400]


def _llm_summary(inputs: dict, output: dict) -> Optional[str]:
    if not _USE_OPENAI or _openai_client is None:
        return None
    try:
        prompt = "Summarize the salient facts and outcomes in one sentence."
        content = json.dumps({"inputs": inputs, "output": output})[:6000]
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": f"{prompt}\n\n{content}"}]
        )
        return resp.choices[0].message.content.strip()[:400]
    except Exception:
        return None


# ------- Embeddings (OpenAI or deterministic fallback) -------
def _cheap_hash_embed(text: str, dim: int = 64) -> List[float]:
    import hashlib as _hl
    vec = [0.0] * dim
    b = text.encode("utf-8", errors="ignore")
    if not b:
        return [1.0] + [0.0] * (dim - 1)
    for i in range(0, len(b), 32):
        h = _hl.sha256(b[i:i+32]).digest()
        for j in range(dim):
            vec[j] += h[j % len(h)] / 255.0
    norm = (sum(x*x for x in vec) ** 0.5) or 1.0
    return [x / norm for x in vec]


def _embed_text(text: str) -> List[float]:
    if _USE_OPENAI and _openai_client is not None:
        try:
            resp = _openai_client.embeddings.create(
                model=_EMBED_MODEL,
                input=text[:7000]
            )
            v = resp.data[0].embedding
            norm = (sum(x*x for x in v) ** 0.5) or 1.0
            return [x / norm for x in v]
        except Exception:
            pass
    return _cheap_hash_embed(text)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x*y for x, y in zip(a, b)))


class GraphMemory:
    def __init__(self,
                 uri: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 token_budget: int = _MEM_TOKEN_BUDGET):
        uri = uri or _NEO4J_URI
        user = user or _NEO4J_USER
        password = password or _NEO4J_PASSWORD
        if not all([uri, user, password]):
            raise RuntimeError("NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD must be set.")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._budget = token_budget

    # ---------- schema ----------
    def init_schema(self):
        stmts = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:ToolCall) REQUIRE t.key IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (t:ToolCall) ON (t.last_access_ts)",
            "CREATE INDEX IF NOT EXISTS FOR (t:ToolCall) ON (t.created_ts)",
            "CREATE INDEX IF NOT EXISTS FOR (t:ToolCall) ON (t.salience)"
        ]
        with self._driver.session() as s:
            for cy in stmts:
                s.run(cy)

    # ---------- core ops ----------
    def remember_or_get(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        output: Dict[str, Any],
        *,
        prev_id: Optional[str] = None,
        derived_from_ids: Optional[List[str]] = None
    ) -> Tuple[str, bool]:
        key = _stable_key(tool_name, inputs)
        now = _now()
        inputs_json = _canonical_json(inputs)
        output_json = _canonical_json(output)
        token_est = _rough_token_estimate(output_json)

        summary = _llm_summary(inputs, output) or _one_line_summary(tool_name, inputs, output)
        embed_text = f"{summary}\n{inputs_json}"
        emb_vec = _embed_text(embed_text)

        with self._driver.session() as s:
            rec = s.run(
                "MATCH (t:ToolCall {key:$key}) "
                "SET t.last_access_ts=$now, t.hit_count = coalesce(t.hit_count,0)+1 "
                "RETURN elementId(t) as id, t.embedding as embedding",
                key=key, now=now
            ).single()
            if rec:
                node_id = rec["id"]
                if rec.get("embedding") is None:
                    s.run("MATCH (t:ToolCall) WHERE elementId(t)=$id SET t.embedding=$emb",
                          id=node_id, emb=emb_vec)
                self._wire_edges(s, node_id, prev_id, derived_from_ids)
                return node_id, True

            res = s.run(
                """
                CREATE (t:ToolCall {
                    key:$key,
                    tool_name:$tool,
                    inputs_json:$inputs_json,
                    created_ts:$now,
                    last_access_ts:$now,
                    hit_count:0,
                    compressed:false,
                    summary:$summary,
                    token_estimate:$tok,
                    embedding:$emb
                })
                CREATE (b:Blob {json:$out_json})
                CREATE (t)-[:HAS_BLOB]->(b)
                RETURN elementId(t) as id
                """,
                key=key, tool=tool_name, inputs_json=inputs_json, now=now,
                summary=summary, tok=token_est, emb=emb_vec, out_json=output_json
            ).single()
            node_id = res["id"]
            self._wire_edges(s, node_id, prev_id, derived_from_ids)
            self.compress_over_budget(session=s)
            return node_id, False

    def _wire_edges(self, session, node_id: str,
                    prev_id: Optional[str],
                    derived_from_ids: Optional[List[str]]):
        if prev_id:
            session.run(
                """
                MATCH (a:ToolCall) WHERE elementId(a)=$prev
                MATCH (b:ToolCall) WHERE elementId(b)=$cur
                MERGE (a)-[:NEXT]->(b)
                """,
                prev=prev_id, cur=node_id
            )
        if derived_from_ids:
            session.run(
                """
                UNWIND $ids as rid
                MATCH (a:ToolCall) WHERE elementId(a)=rid
                MATCH (b:ToolCall) WHERE elementId(b)=$cur
                MERGE (b)-[:DERIVED_FROM]->(a)
                """,
                ids=list(derived_from_ids), cur=node_id
            )

    # ---------- compression / expansion ----------
    def compress_over_budget(self, session=None):
        close_after = False
        if session is None:
            session = self._driver.session()
            close_after = True

        total = session.run(
            "MATCH (t:ToolCall {compressed:false}) "
            "RETURN coalesce(sum(t.token_estimate),0) as total"
        ).single()["total"]

        if total <= self._budget:
            if close_after:
                session.close()
            return

        rows = session.run(
            """
            MATCH (t:ToolCall {compressed:false})
            RETURN elementId(t) as id, t.token_estimate as tok
            ORDER BY t.last_access_ts ASC
            """
        ).data()

        acc = int(total)
        ids = []
        for r in rows:
            if acc <= self._budget:
                break
            ids.append(r["id"])
            acc -= int(r.get("tok") or 0)

        if ids:
            session.run(
                "MATCH (t:ToolCall) WHERE elementId(t) IN $ids SET t.compressed=true",
                ids=ids
            )
        if close_after:
            session.close()

    def compress_nodes(self, node_ids: List[str]):
        with self._driver.session() as s:
            s.run(
                "MATCH (t:ToolCall) WHERE elementId(t) IN $ids SET t.compressed=true",
                ids=list(node_ids)
            )

    def expand(self, node_id: str) -> Dict[str, Any]:
        with self._driver.session() as s:
            rec = s.run(
                """
                MATCH (t:ToolCall)-[:HAS_BLOB]->(b:Blob)
                WHERE elementId(t)=$id
                RETURN b.json as json
                """,
                id=node_id
            ).single()
            if not rec:
                raise KeyError(f"No blob for node {node_id}")
            return json.loads(rec["json"])

    # ---------- salience ----------
    def top_salient(self, limit: int = 10) -> List[Dict[str, Any]]:
        now = _now()
        with self._driver.session() as s:
            rows = s.run(
                """
                MATCH (t:ToolCall)
                RETURN elementId(t) as id, t.tool_name as tool, t.summary as summary,
                       t.last_access_ts as last_access_ts, t.hit_count as hit_count
                """
            ).data()

        scored = []
        for r in rows:
            age_min = max(0.0, (now - float(r["last_access_ts"])) / 60.0)
            recency_score = math.exp(-age_min / 60.0)
            popularity = math.log1p(int(r.get("hit_count", 0)))
            salience = 0.6 * recency_score + 0.4 * popularity
            scored.append({
                "id": r["id"],
                "tool": r["tool"],
                "summary": r.get("summary"),
                "hit_count": int(r.get("hit_count", 0)),
                "salience": float(salience)
            })
        scored.sort(key=lambda x: x["salience"], reverse=True)
        return scored[:limit]

    # ---------- NEW: temporal decay + adaptive compression ----------
    def decay_and_compress(self,
                           half_life_min: float = 60.0,
                           w_recency: float = 0.6,
                           w_popularity: float = 0.4,
                           cutoff: float = 0.2) -> Dict[str, int | float]:
        """
        Recompute t.salience using exponential temporal decay + popularity, then
        compress any node below 'cutoff'.

        salience = w_recency * exp(- age_min / half_life_min)
                 + w_popularity * log(1 + hit_count)
        """
        now = _now()
        with self._driver.session() as s:
            # 1) recompute salience in-place (pure Cypher)
            s.run(
                """
                WITH $half AS half, $wr AS wr, $wp AS wp, $now AS now
                MATCH (t:ToolCall)
                WITH t, half, wr, wp, now,
                     CASE
                       WHEN t.last_access_ts IS NULL THEN 0.0
                       ELSE (now - t.last_access_ts) / 60.0
                     END AS age_min
                WITH t,
                     exp( - age_min / half ) AS recency,
                     log(1 + coalesce(t.hit_count,0)) AS popularity,
                     wr, wp
                SET t.salience = wr * recency + wp * popularity
                """,
                half=half_life_min, wr=w_recency, wp=w_popularity, now=now
            )
            # 2) compress below threshold
            res = s.run(
                """
                MATCH (t:ToolCall)
                WHERE coalesce(t.salience, 0.0) < $cut AND t.compressed = false
                SET t.compressed = true
                RETURN count(t) AS n
                """,
                cut=cutoff
            ).single()
            compressed = int(res["n"]) if res and "n" in res else 0
        return {
            "compressed": compressed,
            "cutoff": float(cutoff),
            "half_life_min": float(half_life_min),
            "w_recency": float(w_recency),
            "w_popularity": float(w_popularity),
        }

    # ---------- utils ----------
    def get(self, node_id: str) -> Dict[str, Any]:
        with self._driver.session() as s:
            rec = s.run(
                """
                MATCH (t:ToolCall)
                WHERE elementId(t)=$id
                OPTIONAL MATCH (t)-[:HAS_BLOB]->(b:Blob)
                RETURN t as t, b.json as blob, elementId(t) as id
                """,
                id=node_id
            ).single()
            if not rec:
                raise KeyError(f"ToolCall {node_id} not found")
            t = rec["t"]
            data = dict(t)
            data["id"] = rec["id"]
            data["has_blob"] = rec["blob"] is not None
            return data

    def backfill_missing_embeddings(self, batch: int = 200):
        with self._driver.session() as s:
            rows = s.run(
                """
                MATCH (t:ToolCall)
                WHERE t.embedding IS NULL
                RETURN elementId(t) as id, t.summary as summary, t.inputs_json as inputs
                LIMIT $lim
                """, lim=batch
            ).data()
            for r in rows:
                text = f"{r.get('summary','')}\n{r.get('inputs','')}"
                emb = _embed_text(text)
                s.run("MATCH (t:ToolCall) WHERE elementId(t)=$id SET t.embedding=$emb", id=r["id"], emb=emb)

    def hybrid_recall(self,
                      anchor_id: Optional[str] = None,
                      query_text: Optional[str] = None,
                      *,
                      hops: int = 3,
                      k: int = 5,
                      include_next: bool = True,
                      include_derived: bool = True) -> List[Dict[str, Any]]:
        with self._driver.session() as s:
            candidates = []

            if anchor_id is not None:
                parts = []
                if include_derived:
                    parts.append(f"""
                    MATCH (a:ToolCall) WHERE elementId(a)=$id
                    MATCH (a)-[:DERIVED_FROM*1..{hops}]->(cand:ToolCall)
                    RETURN DISTINCT elementId(cand) as id
                    """)
                    parts.append(f"""
                    MATCH (a:ToolCall) WHERE elementId(a)=$id
                    MATCH (cand:ToolCall)-[:DERIVED_FROM*1..{hops}]->(a)
                    RETURN DISTINCT elementId(cand) as id
                    """)
                if include_next:
                    parts.append("""
                    MATCH (a:ToolCall) WHERE elementId(a)=$id
                    MATCH (a)-[:NEXT]->(cand:ToolCall)
                    RETURN DISTINCT elementId(cand) as id
                    """)
                    parts.append("""
                    MATCH (a:ToolCall) WHERE elementId(a)=$id
                    MATCH (cand:ToolCall)-[:NEXT]->(a)
                    RETURN DISTINCT elementId(cand) as id
                    """)

                if parts:
                    cy = " UNION ".join(parts)
                    ids = [row["id"] for row in s.run(cy, id=anchor_id).data()]
                    if ids:
                        rows = s.run(
                            """
                            MATCH (t:ToolCall)
                            WHERE elementId(t) IN $ids AND t.embedding IS NOT NULL
                            RETURN elementId(t) as id, t.tool_name as tool, t.summary as summary, t.embedding as emb,
                                   t.hit_count as hit_count, t.compressed as compressed
                            """,
                            ids=ids
                        ).data()
                        candidates = rows

            if not candidates:
                rows = s.run(
                    """
                    MATCH (t:ToolCall)
                    WHERE t.embedding IS NOT NULL
                    RETURN elementId(t) as id, t.tool_name as tool, t.summary as summary, t.embedding as emb,
                           t.hit_count as hit_count, t.compressed as compressed
                    """
                ).data()
                candidates = rows

            if query_text is None and anchor_id is not None:
                anchor = s.run(
                    "MATCH (t:ToolCall) WHERE elementId(t)=$id RETURN t.summary as summary, t.inputs_json as inputs",
                    id=anchor_id
                ).single()
                query_text = f"{anchor['summary']}\n{anchor['inputs']}" if anchor else ""

        qvec = _embed_text(query_text or "")

        ranked = []
        for r in candidates:
            sim = _cosine(qvec, r.get("emb") or [])
            ranked.append({
                "id": r["id"],
                "tool": r.get("tool"),
                "summary": r.get("summary"),
                "hit_count": int(r.get("hit_count") or 0),
                "compressed": bool(r.get("compressed")),
                "cosine": sim
            })
        ranked.sort(key=lambda x: x["cosine"], reverse=True)
        return ranked[:k]

    def close(self):
        self._driver.close()
