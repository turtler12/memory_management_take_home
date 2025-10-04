# src/graph_dashboard.py
from __future__ import annotations
import os, math, json, time
from typing import List, Dict, Any, Tuple
import streamlit as st
from streamlit.components.v1 import html
from neo4j import GraphDatabase
from pyvis.network import Network
from datetime import datetime

# ---------- helpers ----------
def get_session():
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    pwd = os.environ.get("NEO4J_PASSWORD")
    if not (uri and user and pwd):
        st.error("Please set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in your environment.")
        st.stop()
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    return driver.session()

def to_dt(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"

def compute_salience(last_access_ts: float, hit_count: int, now_ts: float) -> float:
    age_min = max(0.0, (now_ts - float(last_access_ts or now_ts)) / 60.0)
    recency = math.exp(-age_min / 60.0)
    popularity = math.log1p(int(hit_count or 0))
    return 0.6 * recency + 0.4 * popularity

def fetch_all_nodes(session, limit: int) -> List[Dict[str, Any]]:
    cy = """
    MATCH (t:ToolCall)
    RETURN elementId(t) as id, t.tool_name as tool, t.summary as summary, t.compressed as compressed,
           t.last_access_ts as last_access_ts, t.hit_count as hit_count, t.token_estimate as token_estimate,
           t.salience as salience
    ORDER BY t.last_access_ts DESC
    LIMIT $limit
    """
    return session.run(cy, limit=limit).data()

def fetch_subgraph_nodes(session, anchor_id: str, hops: int, limit: int,
                         include_next: bool, include_derived: bool) -> List[Dict[str, Any]]:
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

    if not parts:
        ids = [anchor_id]
    else:
        cy_ids = " UNION ".join(parts)
        ids = [r["id"] for r in session.run(cy_ids, id=anchor_id).data()]
        ids = list({*ids, anchor_id})

    if not ids:
        return []

    cy = """
    MATCH (t:ToolCall)
    WHERE elementId(t) IN $ids
    RETURN elementId(t) as id, t.tool_name as tool, t.summary as summary, t.compressed as compressed,
           t.last_access_ts as last_access_ts, t.hit_count as hit_count, t.token_estimate as token_estimate,
           t.salience as salience
    LIMIT $limit
    """
    return session.run(cy, ids=ids, limit=limit).data()

def fetch_edges(session, ids: List[str], include_next: bool, include_derived: bool) -> List[Tuple[str, str, str]]:
    if not ids:
        return []
    edges = []
    if include_next:
        cy_next = """
        MATCH (a:ToolCall)-[:NEXT]->(b:ToolCall)
        WHERE elementId(a) IN $ids AND elementId(b) IN $ids
        RETURN elementId(a) as src, elementId(b) as dst
        """
        edges += [(r["src"], r["dst"], "NEXT") for r in session.run(cy_next, ids=ids).data()]
    if include_derived:
        cy_d = """
        MATCH (a:ToolCall)<-[:DERIVED_FROM]-(b:ToolCall)
        WHERE elementId(a) IN $ids AND elementId(b) IN $ids
        RETURN elementId(b) as src, elementId(a) as dst
        """
        edges += [(r["src"], r["dst"], "DERIVED_FROM") for r in session.run(cy_d, ids=ids).data()]
    return edges

def build_network(nodes: List[Dict[str, Any]],
                  edges: List[Tuple[str, str, str]],
                  physics: bool) -> str:
    net = Network(height="750px", width="100%", directed=True, bgcolor="#0b1220", font_color="#e5e7eb")
    net.toggle_physics(physics)
    COLOR_ACTIVE = "#4f46e5"
    COLOR_COMP   = "#9ca3af"

    now = time.time()
    for n in nodes:
        # prefer DB salience if present; fall back to client calc
        sal = n.get("salience")
        if sal is None:
            sal = compute_salience(n.get("last_access_ts", now), n.get("hit_count", 0), now)
        size = 10 + 35 * min(max(float(sal), 0.0), 1.5) / 1.5
        color = COLOR_COMP if n.get("compressed") else COLOR_ACTIVE
        title = (
            f"<b>{n.get('tool')}</b><br/>"
            f"{(n.get('summary') or '')}<br/><br/>"
            f"<b>last_access:</b> {to_dt(n.get('last_access_ts'))}<br/>"
            f"<b>hit_count:</b> {n.get('hit_count')}<br/>"
            f"<b>token_estimate:</b> {n.get('token_estimate')}<br/>"
            f"<b>salience:</b> {round(float(sal),4)}"
        )
        net.add_node(
            n["id"],
            label=n.get("tool") or "ToolCall",
            title=title,
            color=color,
            size=size
        )

    for src, dst, kind in edges:
        net.add_edge(src, dst, label=kind, color="#94a3b8")

    options = {
        "nodes": {"shape": "dot", "borderWidth": 1, "shadow": True},
        "edges": {"arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
                  "smooth": {"enabled": True, "type": "dynamic"}},
        "physics": {"stabilization": {"iterations": 200}},
        "interaction": {"hover": True, "tooltipDelay": 120}
    }
    net.set_options(json.dumps(options))
    return net.generate_html()

# ---------- UI ----------
st.set_page_config(page_title="A37 Graph Memory Dashboard", layout="wide")
st.title("Graph-Vector Memory Dashboard")
st.caption("Nodes sized by salience; gray = compressed. Decay & compress can be triggered below.")

with st.sidebar:
    st.header("Query")
    mode = st.radio("Mode", ["All nodes", "Subgraph around anchor"], index=0)
    node_limit = st.slider("Node limit", 20, 1000, 200, step=20)
    include_next = st.checkbox("Include NEXT edges", True)
    include_derived = st.checkbox("Include DERIVED_FROM edges", True)
    hops =  st.slider("Hops (anchor subgraph)", 1, 5, 3)
    physics = st.checkbox("Enable physics", True)
    anchor_id = ""
    if mode == "Subgraph around anchor":
        anchor_id = st.text_input("Anchor elementId(t)", value="")
    st.divider()
    st.header("Decay & Compress (server-side)")
    half_life = st.number_input("Half-life (minutes)", min_value=1.0, max_value=1440.0, value=60.0, step=1.0)
    w_recency = st.slider("Weight: recency", 0.0, 1.0, 0.6, 0.05)
    w_pop = st.slider("Weight: popularity", 0.0, 1.0, 0.4, 0.05)
    cutoff = st.slider("Compress below salience", 0.0, 2.0, 0.2, 0.01)
    run_decay = st.button("Run decay & compress now")

session = get_session()
try:
    # Trigger server-side decay/compress if requested
    if run_decay:
        from .graph_memory import GraphMemory  # local import to reuse env
        gm = GraphMemory()
        stats = gm.decay_and_compress(half_life_min=float(half_life),
                                      w_recency=float(w_recency),
                                      w_popularity=float(w_pop),
                                      cutoff=float(cutoff))
        gm.close()
        st.success(f"Compressed {stats['compressed']} nodes (cutoff={stats['cutoff']}, half_life={stats['half_life_min']}m).")
        st.experimental_rerun()

    # Fetch nodes/edges for visualization
    if mode == "All nodes":
        nodes = fetch_all_nodes(session, node_limit)
    else:
        if not anchor_id:
            st.warning("Enter an anchor elementId to fetch a subgraph.")
            nodes = []
        else:
            nodes = fetch_subgraph_nodes(session, anchor_id, hops, node_limit, include_next, include_derived)

    ids = [n["id"] for n in nodes]
    edges = fetch_edges(session, ids, include_next, include_derived) if ids else []

    col1, col2 = st.columns([5, 2], gap="large")
    with col1:
        html(build_network(nodes, edges, physics), height=770, scrolling=True)
    with col2:
        st.subheader("Stats")
        st.json({"nodes": len(nodes), "edges": len(edges), "mode": mode, "hops": hops if mode != "All nodes" else None})
        if nodes:
            st.subheader("Top (by salience)")
            now = time.time()
            ranked = []
            for n in nodes:
                sal = n.get("salience")
                if sal is None:
                    sal = compute_salience(n.get("last_access_ts", now), n.get("hit_count", 0), now)
                ranked.append((float(sal), {"id": n["id"], "tool": n["tool"], "hit_count": n["hit_count"], "last_access": to_dt(n["last_access_ts"]) }))
            ranked.sort(key=lambda x: x[0], reverse=True)
            st.json([r[1] | {"salience": round(float(r[0]), 4)} for r in ranked[:10]])

finally:
    session.close()
