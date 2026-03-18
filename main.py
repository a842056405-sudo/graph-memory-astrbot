from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import os

from .graph_engine import GraphMemoryEngine
from .compactor import GraphCompactor


@register("astrbot_plugin_graph_memory", "adoresever",
          "知识图谱记忆插件 - 从对话中提取结构化知识，实现跨会话记忆", "1.0.0",
          "https://github.com/adoresever/graph-memory")
class GraphMemoryPlugin(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config or {}
        self.compact_threshold = self._get_int_config("compact_turn_count", "GM_COMPACT_TURN_COUNT", 6)
        self.recall_max_nodes = self._get_int_config("recall_max_nodes", "GM_RECALL_MAX_NODES", 6)
        self.recall_max_depth = self._get_int_config("recall_max_depth", "GM_RECALL_MAX_DEPTH", 2)
        self.pagerank_damping = self._get_float_config("pagerank_damping", "GM_PAGERANK_DAMPING", 0.85)
        self.pagerank_iterations = self._get_int_config("pagerank_iterations", "GM_PAGERANK_ITERATIONS", 20)
        self.vector_sync_limit = self._get_int_config("vector_sync_limit", "GM_VECTOR_SYNC_LIMIT", 40)
        self.vector_dedup_threshold = self._get_float_config("vector_dedup_threshold", "GM_VECTOR_DEDUP_THRESHOLD", 0.90)
        self.vector_dedup_max_pairs = self._get_int_config("vector_dedup_max_pairs", "GM_VECTOR_DEDUP_MAX_PAIRS", 20)
        self.engine = GraphMemoryEngine(
            recall_max_nodes=self.recall_max_nodes,
            recall_max_depth=self.recall_max_depth,
            pagerank_damping=self.pagerank_damping,
            pagerank_iterations=self.pagerank_iterations,
        )
        self.compactor = GraphCompactor(self.engine)
        self.msg_counts = {}
        self.turn_map = {}
        self.compact_running = {}
        self.compact_abort = {}
        logger.info("[GraphMemory] 插件已加载")

    def _get_int_config(self, key: str, env_key: str, default: int) -> int:
        try:
            if key in self.config:
                return int(self.config.get(key))
        except Exception:
            pass
        return int(os.getenv(env_key, str(default)))

    def _get_float_config(self, key: str, env_key: str, default: float) -> float:
        try:
            if key in self.config:
                return float(self.config.get(key))
        except Exception:
            pass
        return float(os.getenv(env_key, str(default)))

    def _get_embedding_provider(self):
        try:
            providers = self.context.get_all_embedding_providers()
            if providers:
                return providers[0]
        except Exception:
            return None
        return None

    async def _embed_query(self, text: str):
        ep = self._get_embedding_provider()
        if not ep or not text:
            return None
        try:
            return await ep.get_embedding(text)
        except Exception:
            return None

    async def _sync_vectors_and_dedup(self):
        ep = self._get_embedding_provider()
        if not ep:
            return {"embedded": 0, "merged": 0}
        nodes = self.engine.get_nodes_need_embedding(limit=self.vector_sync_limit)
        if not nodes:
            dedup = self.engine.dedup_by_vectors(
                threshold=self.vector_dedup_threshold,
                max_pairs=self.vector_dedup_max_pairs,
            )
            return {"embedded": 0, "merged": dedup.get("merged", 0)}
        texts = [f"{n['name']}: {n['description']}\n{n['content'][:500]}" for n in nodes]
        embedded = 0
        try:
            if hasattr(ep, "get_embeddings_batch"):
                vectors = await ep.get_embeddings_batch(texts, batch_size=16, tasks_limit=3)
            else:
                vectors = []
                for t in texts:
                    vectors.append(await ep.get_embedding(t))
            for n, vec in zip(nodes, vectors):
                if isinstance(vec, list) and vec:
                    self.engine.save_vector(n["id"], n["content_hash"], vec)
                    embedded += 1
        except Exception:
            pass
        dedup = self.engine.dedup_by_vectors(
            threshold=self.vector_dedup_threshold,
            max_pairs=self.vector_dedup_max_pairs,
        )
        return {"embedded": embedded, "merged": dedup.get("merged", 0)}

    def _get_provider(self, event: AstrMessageEvent):
        """获取当前使用的 LLM provider"""
        umo = event.unified_msg_origin
        return self.context.get_using_provider(umo)

    async def _try_compact(self, event: AstrMessageEvent):
        sid = event.unified_msg_origin or "default"
        if self.compact_running.get(sid):
            return
        provider = self._get_provider(event)
        if not provider:
            logger.warning("[GraphMemory] 未找到 LLM provider，跳过压缩")
            return
        self.compact_running[sid] = True
        self.compact_abort[sid] = False
        try:
            result = await self.compactor.run_compaction(
                provider,
                session_id=sid,
                should_abort=lambda: self.compact_abort.get(sid, False),
            )
            if result.get("compacted"):
                vec_stats = await self._sync_vectors_and_dedup()
                logger.info(
                    f"[GraphMemory] 自动压缩完成 sid={sid[:12]} nodes={result.get('nodes', 0)} edges={result.get('edges', 0)} embed={vec_stats.get('embedded',0)} merged={vec_stats.get('merged',0)}"
                )
        finally:
            self.compact_running[sid] = False
            self.compact_abort[sid] = False

    def _extract_command_args(self, event: AstrMessageEvent, command: str) -> str:
        msg = (event.message_str or "").strip()
        if not msg:
            return ""
        prefixes = [f"/{command}", command]
        for p in prefixes:
            if msg.startswith(p):
                return msg[len(p):].strip()
        parts = msg.split(maxsplit=1)
        if len(parts) == 2:
            return parts[1].strip()
        return ""

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        try:
            sid = event.unified_msg_origin or "default"
            if self.compact_running.get(sid):
                self.compact_abort[sid] = True
            query = req.prompt or event.message_str
            if not query:
                return
            turn = self.engine.next_turn_index(sid)
            self.turn_map[sid] = turn
            self.engine.record_message(session_id=sid, role="user", content=query, turn_index=turn)
            qvec = await self._embed_query(query)
            rec = self.engine.recall(
                query=query,
                max_nodes=self.recall_max_nodes,
                max_depth=None,
                query_vector=qvec,
            )
            memory_context = self.engine.assemble(query=query, session_id=sid, max_nodes=self.recall_max_nodes)
            if memory_context:
                existing = event.get_extra("system_prompt_addon") or ""
                addon = (
                    "\n\n## Graph Memory — 知识图谱记忆\n"
                    "优先参考下方 <knowledge_graph> 中可复用经验，再回答当前问题。\n\n"
                    f"{memory_context}"
                )
                event.set_extra("system_prompt_addon", existing + addon)
                req.system_prompt = (req.system_prompt or "") + addon
                logger.debug(f"[GraphMemory] 注入了知识图谱上下文")
            if rec.get("nodes"):
                logger.debug(f"[GraphMemory] recall命中 {len(rec.get('nodes', []))} 节点")
        except Exception as e:
            logger.exception(f"[GraphMemory] on_llm_request 处理失败: {e}")

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        try:
            sid = event.unified_msg_origin or "default"
            resp_text = resp.completion_text or event.get_result_message_str()
            if resp_text:
                turn = self.turn_map.get(sid)
                self.engine.record_message(session_id=sid, role="assistant", content=resp_text, turn_index=turn)
            self.msg_counts[sid] = self.msg_counts.get(sid, 0) + 1
            if self.msg_counts[sid] >= self.compact_threshold:
                self.msg_counts[sid] = 0
                import asyncio
                asyncio.create_task(self._try_compact(event))
        except Exception as e:
            logger.exception(f"[GraphMemory] on_llm_response 处理失败: {e}")

    @filter.command("gm_stats")
    async def gm_stats(self, event: AstrMessageEvent):
        stats = self.engine.get_stats()
        by_type = ", ".join([f"{k}:{v}" for k, v in stats.get("by_type", {}).items()]) or "-"
        by_edge = ", ".join([f"{k}:{v}" for k, v in stats.get("by_edge_type", {}).items()]) or "-"
        yield event.plain_result(
            f"📊 知识图谱统计\n"
            f"节点数: {stats['nodes']}\n"
            f"边数: {stats['edges']}\n"
            f"待处理消息: {stats['messages']}\n"
            f"节点类型: {by_type}\n"
            f"边类型: {by_edge}\n"
            f"社区数: {stats.get('communities', 0)}"
        )

    @filter.command("gm_compact")
    async def gm_compact(self, event: AstrMessageEvent):
        provider = self._get_provider(event)
        if not provider:
            yield event.plain_result("❌ 未找到可用的 LLM provider")
            return
        sid = event.unified_msg_origin or "default"
        result = await self.compactor.run_compaction(provider, session_id=sid, should_abort=lambda: False)
        if result.get("compacted"):
            vec_stats = await self._sync_vectors_and_dedup()
            stats = self.engine.get_stats()
            yield event.plain_result(
                f"✅ 压缩完成\n新增节点: {result.get('nodes', 0)}\n新增边: {result.get('edges', 0)}\n向量更新: {vec_stats.get('embedded', 0)}\n去重合并: {vec_stats.get('merged', 0)}\n总节点: {stats['nodes']}\n总边: {stats['edges']}"
            )
        else:
            yield event.plain_result(f"ℹ️ 未执行压缩: {result.get('reason', 'unknown')}")

    @filter.command("gm_search")
    async def gm_search(self, event: AstrMessageEvent):
        query = self._extract_command_args(event, "gm_search")
        if not query:
            yield event.plain_result("用法: /gm_search <关键词>")
            return
        sid = event.unified_msg_origin or "default"
        result = self.engine.assemble(query=query, session_id=sid, max_nodes=10)
        if result:
            yield event.plain_result(f"🔍 搜索结果:\n{result}")
        else:
            yield event.plain_result("未找到相关知识")

    @filter.command("gm_finalize")
    async def gm_finalize(self, event: AstrMessageEvent):
        provider = self._get_provider(event)
        if not provider:
            yield event.plain_result("❌ 未找到可用的 LLM provider")
            return
        sid = event.unified_msg_origin or "default"
        result = await self.compactor.finalize_session(provider, session_id=sid)
        if result.get("finalized"):
            yield event.plain_result(
                f"✅ 终审完成\n升级技能: {result.get('promoted', 0)}\n补充关系: {result.get('new_edges', 0)}\n失效节点: {result.get('invalidations', 0)}"
            )
        else:
            yield event.plain_result(f"ℹ️ 终审未执行: {result.get('reason', 'unknown')}")

    @filter.command("gm_record")
    async def gm_record(self, event: AstrMessageEvent):
        args = self._extract_command_args(event, "gm_record")
        if not args:
            yield event.plain_result("用法: /gm_record <TYPE> <name> | <description> | <content>")
            return
        parts = [p.strip() for p in args.split("|")]
        if len(parts) < 3:
            yield event.plain_result("格式错误，示例: /gm_record SKILL apt-install-libgl1 | 安装libgl1 | apt安装命令与场景")
            return
        head = parts[0].split()
        if len(head) < 2:
            yield event.plain_result("请提供 TYPE 和 name")
            return
        node_type = head[0].upper()
        name = " ".join(head[1:])
        description = parts[1]
        content = parts[2]
        sid = event.unified_msg_origin or "manual"
        node = self.engine.upsert_node(node_type=node_type, name=name, description=description, content=content, session_id=sid)
        yield event.plain_result(f"✅ 已记录: {node['name']} ({node['type']})")

    @filter.command("gm_maintain")
    async def gm_maintain(self, event: AstrMessageEvent):
        result = self.engine.run_maintenance()
        vec_stats = await self._sync_vectors_and_dedup()
        tops = result.get("top_pr", [])
        top_text = "\n".join(
            [f"{i + 1}. {x['name']} ({x['type']}, pr={x['pagerank']:.4f})" for i, x in enumerate(tops)]
        ) or "-"
        yield event.plain_result(
            f"🛠️ 图维护完成\n社区数: {result.get('communities', 0)}\n向量更新: {vec_stats.get('embedded', 0)}\n去重合并: {vec_stats.get('merged', 0)}\nPageRank Top:\n{top_text}"
        )
