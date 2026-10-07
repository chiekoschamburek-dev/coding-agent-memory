# codemem

一个面向 **Agent Memory Challenge（代码赛道）** 的代码记忆服务，实现 **Add / Search** 契约。
系统接收并存储某个仓库的工程轨迹，之后在噪声干扰下检索回答问题所需的紧凑证据。

我们只实现 Add 和 Search；Answer 与 Eval 由平台运行。

## 与通用 RAG 栈的区别

代码赛道考察的是**相关 + 噪声**条件下的检索：所有干扰项都来自同一个仓库，因此共享词汇、
文件路径和编码风格。三个由此产生的结果决定了本系统的设计：

1. **结构优先于规模。** 轨迹里混杂着散文、diff、堆栈、日志、命令、测试和配置。分块沿这些
   边界切分，且绝不在代码围栏的中间行切断，因此一个 diff 或一条 traceback 能完整保留。
2. **标识符才是承重信号。** 文件路径、异常类或符号的区分度远高于句子相似度——而且与嵌入
   不同，当候选池里塞满同仓库干扰项时它不会漂移。标识符按确定性方式抽取并按 IDF 加权，
   因此共享 `src/parser/tokenizer.py` 的证据弱于共享 `IndexError`。
3. **不填满预算本身就是目标。** 平台会把我们排序结果的一个按 token 计数的前缀喂给回答
   模型。用同仓库噪声补齐 100 个槽位会挤掉真正的证据。因此噪声门会提前停止，宁可返回
   `[]` 也不返回干扰项。

## 设计不变量

1. **所有内容生成都发生在 Add 阶段。** Search 只做打分、过滤、排序和格式化已存在的记忆。
   经验卡片在 Add 时生成——那时问题还不存在——因此不可能编码某个答案。Search 从不生成文本。
2. **`user_id` 是唯一的硬隔离边界**，在每条读写路径上强制执行，包括实体通道、时效性通道
   和 FTS 索引。
3. **Add 不会因为增强失败而失败。** 原始文本在任何 LLM 工作之前就已持久化并提交，因此
   超时只会降低质量，不会导致失败。

## 架构

```
Add   validate → idempotency → chunk (deterministic) → extract identifiers
      → persist + commit → [enrich: cards/episodes, optional, cached] → 200

Search  query plan → multi-channel recall → RRF fuse → score + gate
        → assemble evidence pyramid → data[]
```

存储是单个 SQLite 数据库（WAL + FTS5），因此一个容器就是一套完整部署，无需外部服务。

| 层 | 内容 | 用 LLM？ | 用途 |
|---|---|---|---|
| L0 `raw_message` | 消息原文 | 否 | 数据只到手一次；审计与重新分块的基础 |
| L1 `chunk` | 结构感知的片段 | 否 | 检索粒度 |
| L2 `chunk_entity` | 路径、符号、异常、测试、命令、包、issue | 否 | 精确匹配检索信号 |
| L3 `memory` | 可检索单元（chunk，后续是 card / episode） | 仅卡片 | 紧凑、可评判的证据 |
| governance | `entity_timeline`、`repo_profile`、`supersedes` | 否 | 时效性权重、仓库身份、软遗忘（已搭骨架） |

## 快速开始

```bash
pip install -e ".[dev]"
python -m codemem            # 监听 0.0.0.0:8080
```

或用 Docker：

```bash
docker compose -f deploy/docker-compose.yml up --build
```

### 接口

| 方法 | 路径 | 鉴权 | 用途 |
| --- | --- | --- | --- |
| `POST` | `/add` | 是 | 存储消息；仅当记忆可被检索后才返回 200 |
| `POST` | `/search` | 是 | 返回排序后的记忆证据 |
| `GET` | `/health` | 否 | 存活探针；任意 2xx |
| `DELETE` | `/admin/users/{user_id}` | 是 | 抹除某个 user_id 的全部痕迹（留存合规） |

鉴权接受 `Authorization: Bearer <key>`、`Authorization: Token <key>` 或 `X-Api-Key`。
设置 `CODEMEM_API_KEY` 即启用鉴权；不设置则以无鉴权方式运行，规则只允许在公开 smoke 阶段
这样做。

业务错误统一使用 `{"detail": {"reason": "..."}}`。

## 配置

每个可调项都是环境变量，这正是镜像与宿主无关的原因。复制 `.env.example` 作为有注释的起点：

```bash
cp .env.example .env    # 然后编辑；切勿提交该文件
```

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CODEMEM_DATA_DIR` | `./data` | SQLite 位置；生产环境请挂载卷 |
| `CODEMEM_PORT` | `8080` | |
| `CODEMEM_API_KEY` | 未设置 | 未设置 = 无鉴权（仅 smoke） |
| `CODEMEM_LLM_ENABLED` | `false` | 增强是可选的，默认关闭 |
| `CODEMEM_LLM_BASE_URL` | 未设置 | OpenAI 兼容端点 |
| `CODEMEM_LLM_API_KEY` | 未设置 | 建议运行时注入，而不是写进仓库文件 |
| `CODEMEM_LLM_MODEL` | `gpt-4o-mini` | 规则为开源赛道固定该模型 |
| `CODEMEM_DENSE_ENABLED` | `true` | 稠密召回；见下方硬件说明 |
| `CODEMEM_RERANK_ENABLED` | `true` | 交叉编码器重排；实测增益最大的一项 |
| `CODEMEM_EMBED_DEVICE` / `CODEMEM_RERANK_DEVICE` | `auto` | 可见 CUDA 时用 GPU，否则 CPU（相差约 8 倍） |
| `CODEMEM_MAX_EVIDENCE_PER_SESSION` | `5` | 单会话多样性上限；与 `CODEMEM_EVIDENCE_MAX_SESSIONS` 配套调整，见 `eval/README.md` |
| `CODEMEM_MIN_EVIDENCE_SCORE` | `0.15` | 噪声门；用 `eval/` 校准 |
| `CODEMEM_EVIDENCE_BUDGET_TOKENS` | `60000` | 载荷总预算，远低于 117,760 的输入窗口 |
| `CODEMEM_SESSION_SELECT_LLM` | `true` | 会话级两段选择（默认含 fail-safe 回退）；见 `eval/README.md` |
| `CODEMEM_SESSION_SELECT_TIMEOUT_SECONDS` | `10` | 选择调用的硬超时 |
| `CODEMEM_SESSION_SCORE_TOPK` | `1` | 会话分聚合器（2 = top-2 质量，已测代理显著、默认未启） |
| `CODEMEM_SESSION_FEATURE_FUSION` | `false` | 会话特征融合（代理显著、答案不转化，默认未启） |
| `CODEMEM_CLAIM_CHANNEL` / `_FLOOR` / `_CAP` | `false` / `0.45` / `3` | claims-only 侧通道（触达↑但答案↓，默认未启） |
| `CODEMEM_DENSE_ELIGIBLE` | `false` | dense-only 准入（方向正、未过显著线；smoke 窗口 A/B 臂） |

> **部署警告：不要携带本机实验的 `.env` 上 VM。** 任何钉值都会覆盖出厂默认——
> 曾实测 `MAX_EVIDENCE_PER_SESSION=3` 的旧钉值让部署跑在未被当前结论支持的配置上。
> VM 上从 `.env.example` 复制并只填 `CODEMEM_API_KEY` 与中继三件套，其余留默认。

**关于模型 Key。** 服务当前完全不需要模型：分块、标识符抽取、BM25、IDF 匹配与融合都是
确定性且无模型的，因此缺少 Key 不会破坏检索。LLM 增强通道是后续交付项；目前这些配置只被
`scripts/probe_provider.py` 消费，用于在依赖某个端点前确认它是否真的表现得像 `gpt-4o-mini`：

```bash
python scripts/probe_provider.py --base-url "$CODEMEM_LLM_BASE_URL" --api-key "$CODEMEM_LLM_API_KEY"
```

## 测试

```bash
pytest -q
```

覆盖契约（回显的标识符、`data` 始终存在、`top_k` 上限、错误信封）、隔离（含实体通道与时效
性通道）、幂等性、Add 降级、同仓库噪声下的排序，以及并发。

`tests/test_ranking_scale.py` 值得单独说明：它使用几百个同仓库干扰项，因为若干真实缺陷在
小语料下不可见，只在真实规模下才暴露。

### 诊断工具

`scripts/` 里保留了发现这些缺陷时用到的测量脚本——保留它们，是因为调参决策由此测量得出，
而非拍脑袋：

| 脚本 | 回答的问题 |
|---|---|
| `audit_chunk_kinds.py` | 真实轨迹上实际出现哪些 chunk 类型，其中多少标签看起来是错的 |
| `characterize_gate.py` | 在只有一条记忆时，噪声门放行了哪些查询 |
| `characterize_gate_corpus.py` | 同上，但面对几百个同仓库干扰项，并报告相关记忆的**排名** |
| `diagnose_lexical.py` | 每个词的文档频率，以及哪些候选击败了相关项 |
| `diagnose_channel_attribution.py` | 逐通道消融，用于归因某次排序失败 |
| `probe_provider.py` | LLM 端点是否表现得像要求的模型 |
| `loadtest.py` | 并发 Add/Search 下的延迟与正确性 |

## 评测

参见 [`eval/README.md`](eval/README.md)：基于 SWEContextBench 构建的代理检索基准及其结果
（MRR 0.72 对随机基线 0.23，nDCG@10 0.61 对 0.19），包括哪些调参来自实测，以及该基准无法
告诉我们的部分。

## 文档

- [`docs/DESIGN.md`](docs/DESIGN.md) — 数据模型、检索设计，以及每条不变量背后的推理。
- [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) — 与规则的逐条对应，以及 Full 前的八项检查清单。
- [`deploy/README.md`](deploy/README.md) — 公网接入方案（平台要求自建公网可达 API），以及 GPU 镜像。

### 关于可选通道的说明

稠密检索与交叉编码器重排在 CPU 或 GPU 上都能跑，但 GPU 大约快 8 倍，而这个差异决定了稠密
召回是否值回成本：在 GPU 上它相对「仅重排」提升了 nDCG@10（0.659 对 0.647）与 recall@10
（0.713 对 0.691）；在 CPU 上则没有可测量的增益。

两种配置都满足契约——300 次 Add 请求在 GPU 上约 155 s，CPU 上约 850 s，都在单请求 30 分钟
上限之内。设备默认 `auto`，因此每台宿主都能得到合适的行为。`GET /health` 会报告实际加载了
什么，因此静默降级是可见的，而不必从分数下降去反推。

## 署名

第三方组件、其许可证，以及本项目原创工作的披露，见 [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md)。
