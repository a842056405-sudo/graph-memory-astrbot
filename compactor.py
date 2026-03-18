import json
import re
from typing import List, Dict, Optional
from astrbot.api import logger


VALID_NODE_TYPES = {"TASK", "SKILL", "EVENT"}
VALID_EDGE_TYPES = {"USED_SKILL", "SOLVED_BY", "REQUIRES", "PATCHES", "CONFLICTS_WITH"}

EXTRACT_SYS = """你是 graph-memory 知识图谱提取引擎，从 AI Agent 对话中提取可复用的结构化知识三元组（节点+关系）。
仅返回 JSON：{"nodes":[...],"edges":[...]}
节点 type 仅允许 TASK/SKILL/EVENT。
边 type 仅允许 USED_SKILL/SOLVED_BY/REQUIRES/PATCHES/CONFLICTS_WITH。
节点字段：type,name,description,content。
边字段：from,to,type,instruction,condition(可选)。
name 必须稳定、可复用、全小写连字符风格。"""


class GraphCompactor:
    def __init__(self, engine):
        self.engine = engine

    def build_prompt(self, messages: List[tuple], existing_names: List[str]) -> str:
        msg_text = "\n\n---\n\n".join(
            [f"[{str(m[3]).upper()} t={m[2]}]\n{str(m[4])[:800]}" for m in messages]
        )
        return f"<System>\n{EXTRACT_SYS}\n\n<Existing Nodes>\n{', '.join(existing_names) if existing_names else '（无）'}\n\n<Conversation>\n{msg_text}"

    def _extract_json(self, text: str) -> str:
        s = (text or "").strip()
        s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE)
        s = re.sub(r"<think>[\s\S]*$", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^```(?:json)?\s*\n?", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\n?\s*```$", "", s, flags=re.IGNORECASE).strip()
        if s.startswith("{") and s.endswith("}"):
            return s
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last > first:
            return s[first:last + 1]
        return s

    def parse_llm_response(self, text: str) -> Optional[Dict]:
        try:
            payload = json.loads(self._extract_json(text))
            return payload
        except Exception:
            return None

    def _normalize_nodes_edges(self, payload: Dict) -> Dict:
        nodes = []
        for n in payload.get("nodes", []) or []:
            ntype = str(n.get("type") or "").upper().strip()
            name = self.engine.normalize_name(str(n.get("name") or ""))
            description = str(n.get("description") or n.get("desc") or "").strip()
            content = str(n.get("content") or "").strip()
            if not content and (name or description):
                content = f"{name}\n{description}".strip()
            if not (ntype in VALID_NODE_TYPES and name and content):
                continue
            nodes.append(
                {"type": ntype, "name": name, "description": description, "content": content}
            )

        type_by_name = {n["name"]: n["type"] for n in nodes}
        edges = []
        for e in payload.get("edges", []) or []:
            fr = self.engine.normalize_name(str(e.get("from") or e.get("source") or ""))
            to = self.engine.normalize_name(str(e.get("to") or e.get("target") or ""))
            et = str(e.get("type") or e.get("relation") or "").upper().strip()
            instruction = str(e.get("instruction") or "").strip()
            condition = str(e.get("condition") or "").strip() or None
            ft = type_by_name.get(fr)
            tt = type_by_name.get(to)
            if ft == "TASK" and tt == "SKILL":
                et = "USED_SKILL"
            elif ft == "EVENT" and tt == "SKILL":
                et = "SOLVED_BY"
            if not (fr and to and et in VALID_EDGE_TYPES and instruction):
                continue
            edges.append(
                {"from": fr, "to": to, "type": et, "instruction": instruction, "condition": condition}
            )
        return {"nodes": nodes, "edges": edges}

    async def run_compaction(self, provider, session_id: str, limit: int = 50, should_abort=None) -> Dict:
        messages = self.engine.get_unextracted_messages(session_id=session_id, limit=limit)
        if len(messages) < 2:
            return {"ok": True, "compacted": False, "reason": "no messages"}

        existing = []
        stats = self.engine.get_stats()
        if stats.get("nodes", 0) > 0:
            recall = self.engine.recall(query="历史经验", max_nodes=20, max_depth=1)
            existing = [n.get("name", "") for n in (recall.get("nodes", []) or []) if n.get("name")]
        prompt = self.build_prompt(messages=messages, existing_names=existing)
        logger.info(f"[GraphMemory] 开始压缩 sid={session_id[:12]} msgs={len(messages)}")

        try:
            if callable(should_abort) and should_abort():
                return {"ok": False, "compacted": False, "reason": "aborted"}
            resp = await provider.text_chat(prompt=prompt)
            if not resp or not resp.completion_text:
                return {"ok": False, "compacted": False, "reason": "empty llm"}

            if callable(should_abort) and should_abort():
                return {"ok": False, "compacted": False, "reason": "aborted"}
            parsed = self.parse_llm_response(resp.completion_text)
            if not parsed:
                logger.warning("[GraphMemory] 无法解析 LLM 响应")
                return {"ok": False, "compacted": False, "reason": "bad json"}

            normalized = self._normalize_nodes_edges(parsed)
            self.engine.ingest(normalized["nodes"], normalized["edges"], session_id=session_id)
            max_turn = max(m[2] for m in messages)
            self.engine.mark_extracted(session_id=session_id, max_turn_index=max_turn)
            return {
                "ok": True,
                "compacted": True,
                "nodes": len(normalized["nodes"]),
                "edges": len(normalized["edges"]),
                "turn": int(max_turn),
            }
        except Exception as e:
            logger.error(f"[GraphMemory] 压缩失败: {e}")
            return {"ok": False, "compacted": False, "reason": str(e)}

    async def finalize_session(self, provider, session_id: str) -> Dict:
        rec = self.engine.recall(query="session summary", max_nodes=30, max_depth=2)
        nodes = rec.get("nodes", [])
        if not nodes:
            return {"ok": True, "finalized": False, "reason": "no nodes"}
        payload = json.dumps(
            [
                {
                    "id": n.get("id"),
                    "type": n.get("type"),
                    "name": n.get("name"),
                    "description": n.get("description"),
                    "validated_count": n.get("validated_count", 1),
                }
                for n in nodes
            ],
            ensure_ascii=False,
        )
        prompt = (
            "你是图谱终审器。仅返回JSON，格式："
            '{"promotedSkills":[{"type":"SKILL","name":"...","description":"...","content":"..."}],'
            '"newEdges":[{"from":"...","to":"...","type":"USED_SKILL|SOLVED_BY|REQUIRES|PATCHES|CONFLICTS_WITH","instruction":"...","condition":""}],'
            '"invalidations":["node_id"]}'
            f"\n当前会话图谱节点:\n{payload}"
        )
        try:
            resp = await provider.text_chat(prompt=prompt)
            if not resp or not resp.completion_text:
                return {"ok": False, "finalized": False, "reason": "empty llm"}
            parsed = self.parse_llm_response(resp.completion_text)
            if not parsed:
                return {"ok": False, "finalized": False, "reason": "bad json"}
            promoted = parsed.get("promotedSkills", []) or []
            new_edges = parsed.get("newEdges", []) or []
            invalidations = parsed.get("invalidations", []) or []
            for n in promoted:
                ntype = str(n.get("type") or "SKILL").upper()
                if ntype != "SKILL":
                    continue
                name = str(n.get("name") or "")
                description = str(n.get("description") or "")
                content = str(n.get("content") or "")
                if name and content:
                    self.engine.upsert_node(ntype, name, description, content, session_id)
            for e in new_edges:
                fr = str(e.get("from") or "")
                to = str(e.get("to") or "")
                et = str(e.get("type") or "").upper()
                instruction = str(e.get("instruction") or "")
                condition = str(e.get("condition") or "").strip() or None
                if fr and to and et in VALID_EDGE_TYPES and instruction:
                    self.engine.upsert_edge(fr, to, et, instruction, session_id, condition)
            if invalidations:
                self.engine.deprecate_nodes(invalidations)
            return {"ok": True, "finalized": True, "promoted": len(promoted), "new_edges": len(new_edges), "invalidations": len(invalidations)}
        except Exception as e:
            logger.error(f"[GraphMemory] session finalize 失败: {e}")
            return {"ok": False, "finalized": False, "reason": str(e)}
