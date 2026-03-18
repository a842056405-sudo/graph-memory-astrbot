import json
import os
import re
import sqlite3
import time
import uuid
import hashlib
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple


class GraphMemoryEngine:
    def __init__(
        self,
        db_path: Optional[str] = None,
        recall_max_nodes: Optional[int] = None,
        recall_max_depth: Optional[int] = None,
        pagerank_damping: Optional[float] = None,
        pagerank_iterations: Optional[int] = None,
    ):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "graph_memory.db")
        self.db_path = db_path
        if self.db_path != ":memory:":
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        self.recall_max_nodes = int(recall_max_nodes if recall_max_nodes is not None else os.getenv("GM_RECALL_MAX_NODES", "6"))
        self.recall_max_depth = int(recall_max_depth if recall_max_depth is not None else os.getenv("GM_RECALL_MAX_DEPTH", "2"))
        self.pagerank_damping = float(pagerank_damping if pagerank_damping is not None else os.getenv("GM_PAGERANK_DAMPING", "0.85"))
        self.pagerank_iterations = int(pagerank_iterations if pagerank_iterations is not None else os.getenv("GM_PAGERANK_ITERATIONS", "20"))
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _now() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def normalize_name(name: str) -> str:
        s = (name or "").strip().lower().replace("_", "-")
        cleaned = []
        for ch in s:
            ok = ch.isalnum() or ch in "- " or ("\u4e00" <= ch <= "\u9fff")
            if ok:
                cleaned.append(ch)
        s = "".join(cleaned).replace(" ", "-")
        while "--" in s:
            s = s.replace("--", "-")
        return s.strip("-")

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS gm_nodes (id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL UNIQUE, description TEXT NOT NULL DEFAULT '', content TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', validated_count INTEGER NOT NULL DEFAULT 1, source_sessions TEXT NOT NULL DEFAULT '[]', community_id TEXT, pagerank REAL NOT NULL DEFAULT 0, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS gm_edges (id TEXT PRIMARY KEY, from_id TEXT NOT NULL, to_id TEXT NOT NULL, type TEXT NOT NULL, instruction TEXT NOT NULL, condition TEXT, session_id TEXT NOT NULL, created_at INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS gm_messages (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, turn_index INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, extracted INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS gm_vectors (node_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, embedding TEXT NOT NULL)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS ix_gm_msg_session ON gm_messages(session_id, turn_index)")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_gm_nodes_type_status ON gm_nodes(type, status)")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_gm_nodes_community ON gm_nodes(community_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_gm_edges_from ON gm_edges(from_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_gm_edges_to ON gm_edges(to_id)")
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS gm_nodes_fts USING fts5(name, description, content, content=gm_nodes, content_rowid=rowid)"
                )
                conn.execute(
                    "CREATE TRIGGER IF NOT EXISTS gm_nodes_ai AFTER INSERT ON gm_nodes BEGIN INSERT INTO gm_nodes_fts(rowid, name, description, content) VALUES (NEW.rowid, NEW.name, NEW.description, NEW.content); END;"
                )
                conn.execute(
                    "CREATE TRIGGER IF NOT EXISTS gm_nodes_ad AFTER DELETE ON gm_nodes BEGIN INSERT INTO gm_nodes_fts(gm_nodes_fts, rowid, name, description, content) VALUES ('delete', OLD.rowid, OLD.name, OLD.description, OLD.content); END;"
                )
                conn.execute(
                    "CREATE TRIGGER IF NOT EXISTS gm_nodes_au AFTER UPDATE ON gm_nodes BEGIN INSERT INTO gm_nodes_fts(gm_nodes_fts, rowid, name, description, content) VALUES ('delete', OLD.rowid, OLD.name, OLD.description, OLD.content); INSERT INTO gm_nodes_fts(rowid, name, description, content) VALUES (NEW.rowid, NEW.name, NEW.description, NEW.content); END;"
                )
            except Exception:
                pass
            conn.commit()

    def next_turn_index(self, session_id: str) -> int:
        sid = session_id or "default"
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_index), 0) FROM gm_messages WHERE session_id = ?",
                (sid,),
            ).fetchone()
        return int(row[0] or 0) + 1

    def record_message(self, session_id: str, role: str, content: str, turn_index: Optional[int] = None):
        if not content:
            return
        sid = session_id or "default"
        tid = turn_index if turn_index is not None else self.next_turn_index(sid)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO gm_messages (id, session_id, turn_index, role, content, extracted, created_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
                (f"m-{uuid.uuid4().hex[:12]}", sid, int(tid), role or "unknown", content, self._now()),
            )
            conn.commit()

    def get_unextracted_messages(self, session_id: str, limit: int = 50) -> List[Tuple]:
        sid = session_id or "default"
        with self._conn() as conn:
            return conn.execute(
                "SELECT id, session_id, turn_index, role, content FROM gm_messages WHERE session_id = ? AND extracted = 0 ORDER BY turn_index ASC, created_at ASC LIMIT ?",
                (sid, limit),
            ).fetchall()

    def mark_extracted(self, session_id: str, max_turn_index: int):
        sid = session_id or "default"
        with self._conn() as conn:
            conn.execute(
                "UPDATE gm_messages SET extracted = 1 WHERE session_id = ? AND turn_index <= ?",
                (sid, int(max_turn_index)),
            )
            conn.commit()

    def _find_by_name(self, conn, name: str):
        n = self.normalize_name(name)
        return conn.execute("SELECT * FROM gm_nodes WHERE name = ?", (n,)).fetchone()

    def upsert_node(self, node_type: str, name: str, description: str, content: str, session_id: str) -> Dict:
        ntype = (node_type or "TASK").strip().upper()
        nname = self.normalize_name(name)
        now = self._now()
        with self._conn() as conn:
            row = self._find_by_name(conn, nname)
            if row:
                existing_sessions = []
                try:
                    existing_sessions = json.loads(row[7] or "[]")
                except Exception:
                    existing_sessions = []
                merged_sessions = list(dict.fromkeys(existing_sessions + [session_id]))
                new_content = content if len(content or "") >= len(row[4] or "") else (row[4] or "")
                new_desc = description if len(description or "") >= len(row[3] or "") else (row[3] or "")
                conn.execute(
                    "UPDATE gm_nodes SET type = ?, description = ?, content = ?, validated_count = validated_count + 1, source_sessions = ?, updated_at = ? WHERE id = ?",
                    (ntype, new_desc, new_content, json.dumps(merged_sessions, ensure_ascii=False), now, row[0]),
                )
                conn.commit()
                node_id = row[0]
            else:
                node_id = f"n-{uuid.uuid4().hex[:12]}"
                conn.execute(
                    "INSERT INTO gm_nodes (id, type, name, description, content, status, validated_count, source_sessions, community_id, pagerank, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'active', 1, ?, NULL, 0, ?, ?)",
                    (node_id, ntype, nname, description or "", content or "", json.dumps([session_id], ensure_ascii=False), now, now),
                )
                conn.commit()
            out = conn.execute(
                "SELECT id, type, name, description, content, validated_count, community_id, pagerank, updated_at FROM gm_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
        return {
            "id": out[0],
            "type": out[1],
            "name": out[2],
            "description": out[3] or "",
            "content": out[4] or "",
            "validated_count": int(out[5] or 1),
            "community_id": out[6],
            "pagerank": float(out[7] or 0),
            "updated_at": int(out[8] or 0),
        }

    def upsert_edge(
        self,
        from_name: str,
        to_name: str,
        edge_type: str,
        instruction: str,
        session_id: str,
        condition: Optional[str] = None,
    ):
        fn = self.normalize_name(from_name)
        tn = self.normalize_name(to_name)
        et = (edge_type or "USED_SKILL").strip().upper()
        with self._conn() as conn:
            fr = conn.execute("SELECT id FROM gm_nodes WHERE name = ?", (fn,)).fetchone()
            tr = conn.execute("SELECT id FROM gm_nodes WHERE name = ?", (tn,)).fetchone()
            if not fr or not tr:
                return
            ex = conn.execute(
                "SELECT id FROM gm_edges WHERE from_id = ? AND to_id = ? AND type = ?",
                (fr[0], tr[0], et),
            ).fetchone()
            if ex:
                conn.execute("UPDATE gm_edges SET instruction = ?, condition = ? WHERE id = ?", (instruction, condition, ex[0]))
            else:
                conn.execute(
                    "INSERT INTO gm_edges (id, from_id, to_id, type, instruction, condition, session_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (f"e-{uuid.uuid4().hex[:12]}", fr[0], tr[0], et, instruction or "", condition, session_id, self._now()),
                )
            conn.commit()

    def deprecate_nodes(self, node_ids: List[str]):
        if not node_ids:
            return
        with self._conn() as conn:
            for node_id in node_ids:
                conn.execute(
                    "UPDATE gm_nodes SET status = 'deprecated', updated_at = ? WHERE id = ?",
                    (self._now(), str(node_id)),
                )
            conn.commit()

    def ingest(self, nodes: List[Dict], edges: List[Dict], session_id: str):
        for node in nodes or []:
            ntype = (node.get("type") or "").upper()
            name = node.get("name") or ""
            desc = node.get("description") or node.get("desc") or ""
            content = node.get("content") or f"{name}\n{desc}".strip()
            if ntype and name and content:
                self.upsert_node(ntype, name, desc, content, session_id)
        for edge in edges or []:
            fr = edge.get("from") or edge.get("source") or ""
            to = edge.get("to") or edge.get("target") or ""
            et = edge.get("type") or edge.get("relation") or ""
            instruction = edge.get("instruction") or ""
            condition = edge.get("condition")
            if fr and to and et and instruction:
                self.upsert_edge(fr, to, et, instruction, session_id, condition)

    def _search_nodes(self, query: str, limit: int = 8) -> List[Dict]:
        q = (query or "").strip()
        if not q:
            return []
        with self._conn() as conn:
            try:
                rows = conn.execute(
                    "SELECT n.id, n.type, n.name, n.description, n.content, n.validated_count, n.pagerank, n.updated_at, n.community_id FROM gm_nodes_fts f JOIN gm_nodes n ON n.rowid = f.rowid WHERE gm_nodes_fts MATCH ? AND n.status = 'active' ORDER BY rank LIMIT ?",
                    (q, limit),
                ).fetchall()
                if rows:
                    return [
                        {"id": r[0], "type": r[1], "name": r[2], "description": r[3] or "", "content": r[4] or "", "validated_count": int(r[5] or 1), "pagerank": float(r[6] or 0), "updated_at": int(r[7] or 0), "community_id": r[8]}
                        for r in rows
                    ]
            except Exception:
                pass
            tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{1,4}", q.lower())
            tokens = [t for t in tokens if t]
            if not tokens:
                rows = conn.execute(
                    "SELECT id, type, name, description, content, validated_count, pagerank, updated_at, community_id FROM gm_nodes WHERE status = 'active' ORDER BY pagerank DESC, validated_count DESC, updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [
                    {"id": r[0], "type": r[1], "name": r[2], "description": r[3] or "", "content": r[4] or "", "validated_count": int(r[5] or 1), "pagerank": float(r[6] or 0), "updated_at": int(r[7] or 0), "community_id": r[8]}
                    for r in rows
                ]
            where = " OR ".join(["(name LIKE ? OR description LIKE ? OR content LIKE ?)"] * len(tokens))
            likes = []
            for t in tokens:
                likes.extend([f"%{t}%", f"%{t}%", f"%{t}%"])
            rows = conn.execute(
                f"SELECT id, type, name, description, content, validated_count, pagerank, updated_at, community_id FROM gm_nodes WHERE status = 'active' AND ({where}) ORDER BY pagerank DESC, validated_count DESC, updated_at DESC LIMIT ?",
                (*likes, limit),
            ).fetchall()
        result = [
            {"id": r[0], "type": r[1], "name": r[2], "description": r[3] or "", "content": r[4] or "", "validated_count": int(r[5] or 1), "pagerank": float(r[6] or 0), "updated_at": int(r[7] or 0), "community_id": r[8]}
            for r in rows
        ]
        if result:
            return result
        return self._top_nodes(limit=limit)

    def _top_nodes(self, limit: int = 6) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, type, name, description, content, validated_count, pagerank, updated_at, community_id FROM gm_nodes WHERE status = 'active' ORDER BY pagerank DESC, validated_count DESC, updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": r[0], "type": r[1], "name": r[2], "description": r[3] or "", "content": r[4] or "", "validated_count": int(r[5] or 1), "pagerank": float(r[6] or 0), "updated_at": int(r[7] or 0), "community_id": r[8]}
            for r in rows
        ]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            xf = float(x)
            yf = float(y)
            dot += xf * yf
            na += xf * xf
            nb += yf * yf
        if na <= 0 or nb <= 0:
            return -1.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.md5((content or "").encode("utf-8")).hexdigest()

    def get_vector_hash(self, node_id: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute("SELECT content_hash FROM gm_vectors WHERE node_id = ?", (node_id,)).fetchone()
        if not row:
            return None
        return row[0]

    def save_vector(self, node_id: str, content_hash: str, embedding: List[float]):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO gm_vectors (node_id, content_hash, embedding) VALUES (?, ?, ?)",
                (node_id, content_hash, json.dumps(embedding, ensure_ascii=False)),
            )
            conn.commit()

    def get_nodes_need_embedding(self, limit: int = 50) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT n.id, n.type, n.name, n.description, n.content FROM gm_nodes n WHERE n.status = 'active' ORDER BY n.updated_at DESC LIMIT ?",
                (max(1, limit * 3),),
            ).fetchall()
        need = []
        for r in rows:
            cid = r[0]
            content = r[4] or ""
            h = self._content_hash(content)
            old_h = self.get_vector_hash(cid)
            if old_h != h:
                need.append({"id": cid, "type": r[1], "name": r[2], "description": r[3] or "", "content": content, "content_hash": h})
            if len(need) >= limit:
                break
        return need

    def vector_seed_nodes(self, query_vector: List[float], limit: int = 6) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT v.node_id, v.embedding, n.type, n.name, n.description, n.content, n.validated_count, n.pagerank, n.updated_at, n.community_id FROM gm_vectors v JOIN gm_nodes n ON n.id = v.node_id WHERE n.status = 'active'"
            ).fetchall()
        scored = []
        for r in rows:
            try:
                emb = json.loads(r[1] or "[]")
            except Exception:
                emb = []
            s = self._cosine(query_vector, emb)
            if s <= 0:
                continue
            scored.append((s, {"id": r[0], "type": r[2], "name": r[3], "description": r[4] or "", "content": r[5] or "", "validated_count": int(r[6] or 1), "pagerank": float(r[7] or 0), "updated_at": int(r[8] or 0), "community_id": r[9]}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:limit]]

    def merge_nodes(self, keep_id: str, merge_id: str):
        if keep_id == merge_id:
            return
        with self._conn() as conn:
            kr = conn.execute("SELECT content, description, validated_count, source_sessions FROM gm_nodes WHERE id = ?", (keep_id,)).fetchone()
            mr = conn.execute("SELECT content, description, validated_count, source_sessions FROM gm_nodes WHERE id = ?", (merge_id,)).fetchone()
            if not kr or not mr:
                return
            k_content, k_desc, k_count, k_sessions = kr
            m_content, m_desc, m_count, m_sessions = mr
            try:
                ss = list(dict.fromkeys((json.loads(k_sessions or "[]") + json.loads(m_sessions or "[]"))))
            except Exception:
                ss = []
            new_content = k_content if len(k_content or "") >= len(m_content or "") else m_content
            new_desc = k_desc if len(k_desc or "") >= len(m_desc or "") else m_desc
            new_count = int(k_count or 1) + int(m_count or 1)
            conn.execute(
                "UPDATE gm_nodes SET content = ?, description = ?, validated_count = ?, source_sessions = ?, updated_at = ? WHERE id = ?",
                (new_content, new_desc, new_count, json.dumps(ss, ensure_ascii=False), self._now(), keep_id),
            )
            conn.execute("UPDATE gm_edges SET from_id = ? WHERE from_id = ?", (keep_id, merge_id))
            conn.execute("UPDATE gm_edges SET to_id = ? WHERE to_id = ?", (keep_id, merge_id))
            conn.execute("DELETE FROM gm_edges WHERE from_id = to_id")
            dup_rows = conn.execute("SELECT id, from_id, to_id, type FROM gm_edges ORDER BY created_at ASC").fetchall()
            seen = set()
            for rid, fr, to, et in dup_rows:
                k = (fr, to, et)
                if k in seen:
                    conn.execute("DELETE FROM gm_edges WHERE id = ?", (rid,))
                else:
                    seen.add(k)
            conn.execute("UPDATE gm_nodes SET status = 'deprecated', updated_at = ? WHERE id = ?", (self._now(), merge_id))
            conn.commit()

    def dedup_by_vectors(self, threshold: float = 0.90, max_pairs: int = 50) -> Dict:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT v.node_id, v.embedding, n.type, n.validated_count, n.updated_at FROM gm_vectors v JOIN gm_nodes n ON n.id = v.node_id WHERE n.status = 'active'"
            ).fetchall()
        vecs = []
        for r in rows:
            try:
                emb = json.loads(r[1] or "[]")
            except Exception:
                emb = []
            if not emb:
                continue
            vecs.append({"id": r[0], "vec": emb, "type": r[2], "validated_count": int(r[3] or 1), "updated_at": int(r[4] or 0)})
        merged = 0
        pairs = 0
        used = set()
        for i in range(len(vecs)):
            a = vecs[i]
            if a["id"] in used:
                continue
            for j in range(i + 1, len(vecs)):
                b = vecs[j]
                if b["id"] in used:
                    continue
                if a["type"] != b["type"]:
                    continue
                sim = self._cosine(a["vec"], b["vec"])
                if sim < threshold:
                    continue
                pairs += 1
                keep = a
                drop = b
                if (b["validated_count"], b["updated_at"]) > (a["validated_count"], a["updated_at"]):
                    keep = b
                    drop = a
                self.merge_nodes(keep["id"], drop["id"])
                used.add(drop["id"])
                merged += 1
                if pairs >= max_pairs:
                    break
            if pairs >= max_pairs:
                break
        return {"pairs": pairs, "merged": merged}

    def _build_graph(self) -> Tuple[Dict[str, Dict], Dict[str, Dict[str, Dict]]]:
        nodes = {}
        adj = defaultdict(dict)
        with self._conn() as conn:
            for r in conn.execute("SELECT id, type, name, description, content, validated_count, pagerank, updated_at, community_id FROM gm_nodes WHERE status = 'active'"):
                nodes[r[0]] = {"id": r[0], "type": r[1], "name": r[2], "description": r[3] or "", "content": r[4] or "", "validated_count": int(r[5] or 1), "pagerank": float(r[6] or 0), "updated_at": int(r[7] or 0), "community_id": r[8]}
            for r in conn.execute("SELECT id, from_id, to_id, type, instruction, condition FROM gm_edges"):
                if r[1] in nodes and r[2] in nodes:
                    payload = {"id": r[0], "from_id": r[1], "to_id": r[2], "type": r[3], "instruction": r[4] or "", "condition": r[5]}
                    adj[r[1]][r[2]] = payload
                    adj[r[2]][r[1]] = payload
        return nodes, adj

    def _graph_walk(self, seed_ids: List[str], max_depth: int = 2):
        nodes, adj = self._build_graph()
        if not seed_ids:
            return [], []
        visited = set()
        q = deque()
        for sid in seed_ids:
            if sid in nodes:
                visited.add(sid)
                q.append((sid, 0))
        while q:
            cur, d = q.popleft()
            if d >= max_depth:
                continue
            for nb in adj.get(cur, {}).keys():
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
        picked_nodes = [nodes[nid] for nid in visited]
        seen_e = set()
        picked_edges = []
        for a in visited:
            for b, e in adj.get(a, {}).items():
                if b not in visited:
                    continue
                eid = e["id"]
                if eid in seen_e:
                    continue
                seen_e.add(eid)
                picked_edges.append(e)
        return picked_nodes, picked_edges

    def _community_representatives(self, limit: int = 2) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, type, name, description, content, validated_count, pagerank, updated_at, community_id FROM gm_nodes WHERE status = 'active' ORDER BY pagerank DESC, validated_count DESC, updated_at DESC"
            ).fetchall()
        picked = []
        seen_community = set()
        for r in rows:
            cid = r[8]
            if cid:
                if cid in seen_community:
                    continue
                seen_community.add(cid)
            picked.append(
                {"id": r[0], "type": r[1], "name": r[2], "description": r[3] or "", "content": r[4] or "", "validated_count": int(r[5] or 1), "pagerank": float(r[6] or 0), "updated_at": int(r[7] or 0), "community_id": cid}
            )
            if len(picked) >= limit:
                break
        return picked

    @staticmethod
    def _merge_recall(precise: Dict, generalized: Dict) -> Dict:
        node_map = {}
        edge_map = {}
        for n in precise.get("nodes", []):
            node_map[n["id"]] = n
        for e in precise.get("edges", []):
            edge_map[e["id"]] = e
        for n in generalized.get("nodes", []):
            if n["id"] not in node_map:
                node_map[n["id"]] = n
        final_ids = set(node_map.keys())
        for e in generalized.get("edges", []):
            if e["id"] not in edge_map and e["from_id"] in final_ids and e["to_id"] in final_ids:
                edge_map[e["id"]] = e
        nodes = list(node_map.values())
        edges = list(edge_map.values())
        token_estimate = sum(len(n.get("content", "")) + len(n.get("description", "")) for n in nodes) // 3 + 1
        return {"nodes": nodes, "edges": edges, "token_estimate": token_estimate}

    def _recall_precise(self, query: str, max_nodes: int, max_depth: int, query_vector: Optional[List[float]] = None) -> Dict:
        seeds = []
        if query_vector:
            seeds = self.vector_seed_nodes(query_vector=query_vector, limit=max_nodes)
        if len(seeds) < 2:
            existed = {n["id"] for n in seeds}
            text_seeds = self._search_nodes(query, limit=max_nodes)
            for n in text_seeds:
                if n["id"] not in existed:
                    seeds.append(n)
        if not seeds:
            return {"nodes": [], "edges": [], "token_estimate": 0}
        seed_ids = [s["id"] for s in seeds]
        nodes, edges = self._graph_walk(seed_ids, max_depth=max_depth)
        if not nodes:
            return {"nodes": [], "edges": [], "token_estimate": 0}
        scores = self._personalized_pagerank(
            seed_ids,
            [n["id"] for n in nodes],
            damping=self.pagerank_damping,
            iterations=self.pagerank_iterations,
        )
        nodes.sort(key=lambda n: (scores.get(n["id"], 0), n["validated_count"], n["updated_at"]), reverse=True)
        nodes = nodes[: max_nodes]
        keep = {n["id"] for n in nodes}
        edges = [e for e in edges if e["from_id"] in keep and e["to_id"] in keep]
        token_estimate = sum(len(n.get("content", "")) + len(n.get("description", "")) for n in nodes) // 3 + 1
        return {"nodes": nodes, "edges": edges, "token_estimate": token_estimate}

    def _recall_generalized(self, query: str, max_nodes: int) -> Dict:
        reps = self._community_representatives(limit=3)
        if not reps:
            reps = self._top_nodes(limit=3)
        if not reps:
            return {"nodes": [], "edges": [], "token_estimate": 0}
        seed_ids = [n["id"] for n in reps]
        nodes, edges = self._graph_walk(seed_ids, max_depth=1)
        if not nodes:
            return {"nodes": [], "edges": [], "token_estimate": 0}
        scores = self._personalized_pagerank(
            seed_ids,
            [n["id"] for n in nodes],
            damping=self.pagerank_damping,
            iterations=self.pagerank_iterations,
        )
        nodes.sort(key=lambda n: (scores.get(n["id"], 0), n["updated_at"], n["validated_count"]), reverse=True)
        nodes = nodes[: max_nodes]
        keep = {n["id"] for n in nodes}
        edges = [e for e in edges if e["from_id"] in keep and e["to_id"] in keep]
        token_estimate = sum(len(n.get("content", "")) + len(n.get("description", "")) for n in nodes) // 3 + 1
        return {"nodes": nodes, "edges": edges, "token_estimate": token_estimate}

    def _personalized_pagerank(self, seed_ids: List[str], candidate_ids: List[str], damping: float = 0.85, iterations: int = 20) -> Dict[str, float]:
        nodes, adj = self._build_graph()
        cands = [cid for cid in candidate_ids if cid in nodes]
        if not cands:
            return {}
        teleport = {cid: 0.0 for cid in cands}
        valid_seeds = [sid for sid in seed_ids if sid in teleport]
        if valid_seeds:
            w = 1.0 / len(valid_seeds)
            for sid in valid_seeds:
                teleport[sid] = w
        else:
            w = 1.0 / len(cands)
            for cid in cands:
                teleport[cid] = w
        pr = {cid: 1.0 / len(cands) for cid in cands}
        for _ in range(iterations):
            nxt = {cid: (1.0 - damping) * teleport[cid] for cid in cands}
            for cid in cands:
                neigh = [n for n in adj.get(cid, {}).keys() if n in pr]
                if not neigh:
                    nxt[cid] += damping * pr[cid]
                    continue
                share = damping * pr[cid] / len(neigh)
                for nb in neigh:
                    nxt[nb] += share
            pr = nxt
        return pr

    def recall(self, query: str, max_nodes: Optional[int] = None, max_depth: Optional[int] = None, query_vector: Optional[List[float]] = None) -> Dict:
        mx_nodes = max_nodes if max_nodes is not None else self.recall_max_nodes
        mx_depth = max_depth if max_depth is not None else self.recall_max_depth
        precise = self._recall_precise(query=query, max_nodes=mx_nodes, max_depth=mx_depth, query_vector=query_vector)
        generalized = self._recall_generalized(query=query, max_nodes=mx_nodes)
        merged = self._merge_recall(precise, generalized)
        merged["nodes"].sort(
            key=lambda n: (
                n.get("pagerank", 0),
                n.get("validated_count", 1),
                n.get("updated_at", 0),
            ),
            reverse=True,
        )
        if len(merged["nodes"]) > mx_nodes:
            keep_ids = {n["id"] for n in merged["nodes"][:mx_nodes]}
            merged["nodes"] = merged["nodes"][:mx_nodes]
            merged["edges"] = [e for e in merged["edges"] if e["from_id"] in keep_ids and e["to_id"] in keep_ids]
        return merged

    def _get_by_session(self, session_id: str) -> List[Dict]:
        sid = session_id or "default"
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, type, name, description, content, validated_count, pagerank, updated_at, community_id, source_sessions FROM gm_nodes WHERE status = 'active' ORDER BY updated_at DESC"
            ).fetchall()
        out = []
        for r in rows:
            try:
                ss = json.loads(r[9] or "[]")
            except Exception:
                ss = []
            if sid in ss:
                out.append({"id": r[0], "type": r[1], "name": r[2], "description": r[3] or "", "content": r[4] or "", "validated_count": int(r[5] or 1), "pagerank": float(r[6] or 0), "updated_at": int(r[7] or 0), "community_id": r[8]})
        return out

    def _edges_for_node_ids(self, ids: set) -> List[Dict]:
        if not ids:
            return []
        with self._conn() as conn:
            rows = conn.execute("SELECT id, from_id, to_id, type, instruction, condition FROM gm_edges").fetchall()
        out = []
        for r in rows:
            if r[1] in ids and r[2] in ids:
                out.append({"id": r[0], "from_id": r[1], "to_id": r[2], "type": r[3], "instruction": r[4] or "", "condition": r[5]})
        return out

    @staticmethod
    def _escape_xml(text: str) -> str:
        return (
            (text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def assemble(self, query: str, session_id: str, max_nodes: int = 6) -> str:
        active_nodes = self._get_by_session(session_id)
        rec = self.recall(query, max_nodes=max_nodes, max_depth=2)
        selected = {}
        for n in rec["nodes"]:
            selected[n["id"]] = dict(n, _src="recalled")
        for n in active_nodes:
            selected[n["id"]] = dict(n, _src="active")
        if not selected:
            return ""
        picked_nodes = list(selected.values())
        type_pri = {"SKILL": 3, "TASK": 2, "EVENT": 1}
        picked_nodes.sort(
            key=lambda n: (0 if n["_src"] == "active" else 1, -type_pri.get(n["type"], 0), -n["validated_count"], -n["pagerank"]),
        )
        ids = {n["id"] for n in picked_nodes}
        active_edges = self._edges_for_node_ids({n["id"] for n in active_nodes})
        all_edges = {e["id"]: e for e in rec["edges"]}
        for e in active_edges:
            all_edges[e["id"]] = e
        used_edges = [e for e in all_edges.values() if e["from_id"] in ids and e["to_id"] in ids]
        id_name = {n["id"]: n["name"] for n in picked_nodes}
        parts = ["<knowledge_graph>"]
        by_comm = defaultdict(list)
        no_comm = []
        for n in picked_nodes:
            if n.get("community_id"):
                by_comm[n["community_id"]].append(n)
            else:
                no_comm.append(n)
        for cid, members in by_comm.items():
            parts.append(f'  <community id="{cid}">')
            for n in members:
                tag = n["type"].lower()
                src_attr = ' source="recalled"' if n["_src"] == "recalled" else ""
                parts.append(
                    f'    <{tag} name="{self._escape_xml(n["name"])}" desc="{self._escape_xml(n["description"])}"{src_attr}>\n{self._escape_xml((n["content"] or "").strip())}\n    </{tag}>'
                )
            parts.append("  </community>")
        for n in no_comm:
            tag = n["type"].lower()
            src_attr = ' source="recalled"' if n["_src"] == "recalled" else ""
            parts.append(
                f'  <{tag} name="{self._escape_xml(n["name"])}" desc="{self._escape_xml(n["description"])}"{src_attr}>\n{self._escape_xml((n["content"] or "").strip())}\n  </{tag}>'
            )
        if used_edges:
            parts.append("  <edges>")
            for e in used_edges:
                cond = f' when="{self._escape_xml(e["condition"])}"' if e.get("condition") else ""
                parts.append(
                    f'    <e type="{e["type"]}" from="{self._escape_xml(id_name.get(e["from_id"], e["from_id"]))}" to="{self._escape_xml(id_name.get(e["to_id"], e["to_id"]))}"{cond}>{self._escape_xml(e["instruction"])}</e>'
                )
            parts.append("  </edges>")
        parts.append("</knowledge_graph>")
        return "\n".join(parts)

    def search(self, query: str, session_id: str, max_nodes: int = 6) -> str:
        return self.assemble(query=query, session_id=session_id, max_nodes=max_nodes)

    def run_maintenance(self) -> Dict:
        nodes, adj = self._build_graph()
        node_ids = list(nodes.keys())
        if not node_ids:
            return {"communities": 0, "top_pr": []}
        scores = self._personalized_pagerank(node_ids[: min(5, len(node_ids))], node_ids)
        with self._conn() as conn:
            for nid, score in scores.items():
                conn.execute("UPDATE gm_nodes SET pagerank = ? WHERE id = ?", (float(score), nid))
            visited = set()
            community_index = 0
            for nid in node_ids:
                if nid in visited:
                    continue
                community_index += 1
                cid = f"c{community_index}"
                dq = deque([nid])
                visited.add(nid)
                members = []
                while dq:
                    x = dq.popleft()
                    members.append(x)
                    for nb in adj.get(x, {}).keys():
                        if nb not in visited:
                            visited.add(nb)
                            dq.append(nb)
                for mid in members:
                    conn.execute("UPDATE gm_nodes SET community_id = ? WHERE id = ?", (cid, mid))
            conn.commit()
            top_rows = conn.execute(
                "SELECT name, type, pagerank FROM gm_nodes WHERE status = 'active' ORDER BY pagerank DESC LIMIT 5"
            ).fetchall()
        return {"communities": community_index, "top_pr": [{"name": r[0], "type": r[1], "pagerank": float(r[2] or 0)} for r in top_rows]}

    def get_stats(self) -> Dict:
        with self._conn() as conn:
            total_nodes = int(conn.execute("SELECT COUNT(*) FROM gm_nodes WHERE status = 'active'").fetchone()[0] or 0)
            total_edges = int(conn.execute("SELECT COUNT(*) FROM gm_edges").fetchone()[0] or 0)
            pending = int(conn.execute("SELECT COUNT(*) FROM gm_messages WHERE extracted = 0").fetchone()[0] or 0)
            by_type_rows = conn.execute("SELECT type, COUNT(*) FROM gm_nodes WHERE status = 'active' GROUP BY type").fetchall()
            by_edge_rows = conn.execute("SELECT type, COUNT(*) FROM gm_edges GROUP BY type").fetchall()
            communities = int(conn.execute("SELECT COUNT(DISTINCT community_id) FROM gm_nodes WHERE status = 'active' AND community_id IS NOT NULL AND community_id <> ''").fetchone()[0] or 0)
        return {
            "nodes": total_nodes,
            "edges": total_edges,
            "messages": pending,
            "by_type": {r[0]: int(r[1]) for r in by_type_rows},
            "by_edge_type": {r[0]: int(r[1]) for r in by_edge_rows},
            "communities": communities,
        }
