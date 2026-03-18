# Graph Memory (AstrBot 适配版)

> 基于 [adoresever/graph-memory](https://github.com/adoresever/graph-memory) 思路实现的 **AstrBot 插件适配版本**。\
> 目标：把长对话压缩为结构化知识图谱，并在后续对话中自动召回，降低上下文膨胀和跨会话遗忘。

## 项目说明

- 上游项目是 OpenClaw 插件（TypeScript + OpenClaw 生命周期槽位）。
- 本项目是 AstrBot 插件（Python + AstrBot 事件钩子）。
- 核心能力保持一致方向：
  - 对话消息入库
  - LLM 抽取三元组（TASK / SKILL / EVENT）
  - 图谱召回注入 `system_prompt`
  - PageRank / 社区维护
  - 向量检索与相似去重（依赖 AstrBot Embedding Provider）

## 核心能力

- **跨会话记忆召回**：在 `on_llm_request` 阶段将图谱知识拼装为 XML 注入系统提示。
- **自动压缩**：达到阈值后异步触发压缩，提取节点与边。
- **双路径召回**：精确路径 + 泛化路径合并，减少空召回。
- **向量增强**：可用 embedding 时执行向量召回与节点去重。
- **会话级并发控制**：同会话压缩互斥，支持请求到来时中断压缩。
- **手动维护能力**：支持手动压缩、终审、维护、搜索、录入。

## 数据库

默认数据库：`graph_memory.db`（位于插件目录）

主要表：

- `gm_nodes`：知识节点（含 `pagerank`、`community_id`、`status`）
- `gm_edges`：知识边（关系类型 + 指令）
- `gm_messages`：原始消息（含 `session_id`、`turn_index`、`extracted`）
- `gm_nodes_fts`：FTS5 全文索引
- `gm_vectors`：节点向量（可选）

## 安装与启用（AstrBot）

1. 将插件目录放入 AstrBot 插件路径（目录名：`astrbot_plugin_graph_memory`）。
2. 确保目录内包含：
   - `main.py`
   - `graph_engine.py`
   - `compactor.py`
   - `metadata.yaml`
   - `__init__.py`
3. 重启 AstrBot，确认日志出现 `"[GraphMemory] 插件已加载"`。

## Embedding 配置（可选但推荐）

本插件优先使用 AstrBot 已配置的 Embedding Provider：

- 已配置embedding：启用向量召回 + 去重
- 未配置 embedding：自动退化为 FTS/关键词检索

### 在 AstrBot 中添加 Embedding 模型教程

1. 打开 AstrBot 管理界面，进入模型或 Provider 管理页面。
2. 新增一个 Provider，类型选择 **嵌入（Embedding）**。
3. 按你使用的平台填写配置：
   - `API Key`
   - `Base URL`（如有）
   - `Model`（例如 `text-embedding-3-small`）
4. 保存配置并重启 AstrBot。
5. 回到对话中执行 `/gm_compact` 或 `/gm_maintain`，如果返回里出现“向量更新”或“去重合并”，说明 embedding 已生效。

说明：

- 本插件会自动使用 AstrBot 中已启用的 Embedding Provider，无需在插件内单独再填一份 embedding 配置。
- 如果没有配置 embedding，本插件仍可使用，只是会退化为关键词/FTS 召回。

## 命令

- `/gm_stats`：查看图谱统计
- `/gm_compact`：手动触发压缩
- `/gm_search <关键词>`：搜索图谱
- `/gm_record <TYPE> <name> | <description> | <content>`：手动写入节点
- `/gm_maintain`：执行图维护（社区、PR、向量同步、去重）
- `/gm_finalize`：执行会话终审（技能升级、补边、失效标记）

## 环境变量

- `GM_COMPACT_TURN_COUNT`：触发自动压缩的轮次阈值（默认 `6`）
- `GM_RECALL_MAX_NODES`：每次注入的最大节点数（默认 `6`）
- `GM_RECALL_MAX_DEPTH`：召回图遍历深度（默认 `2`）
- `GM_PAGERANK_DAMPING`：PPR 阻尼系数（默认 `0.85`）
- `GM_PAGERANK_ITERATIONS`：PPR 迭代次数（默认 `20`）

## 插件面板配置（推荐）

本插件已支持 AstrBot 插件齿轮配置面板（`_conf_schema.json`）。

- 你可以在插件管理页点击齿轮直接修改参数，无需手动改环境变量。
- 面板配置优先级高于环境变量；未填写时回退到环境变量默认值。
- 可在面板中配置：
  - 自动压缩阈值
  - 召回最大节点数
  - 召回深度
  - PageRank 阻尼与迭代次数
  - 向量同步上限
  - 向量去重阈值与每轮最大处理对数

## 与上游 OpenClaw 版差异

- AstrBot 版通过 `on_llm_request / on_llm_response` 接入，不使用 OpenClaw 的 `contextEngine` 槽位。
- AstrBot 没有同名 `session_end` 插件钩子，终审能力通过 `/gm_finalize` 命令触发。
- 逻辑与数据结构尽量对齐上游，但以 AstrBot 运行模型为准。

## 注意事项

- 当前节点按归一化名称合并；跨项目同名实体可能混淆。生产场景建议：
  - 每个项目使用独立实例/独立数据库，或
  - 仅沉淀通用 SKILL，避免把强实体长期固化到全局图谱。

## 致谢

- 原始项目与设计灵感：`adoresever/graph-memory`
- 本仓库为 AstrBot 生态适配实现。
