# API 的各字段（代码赛道）

> 本项目参加**代码赛道**。本文件是官方 API Guide 的代码赛道精简版（2026-10-10 核对，
> 与官方页字段级一致）；多模态赛道的 ContentPart[] 格式见官方页，本项目不使用。

## Add 请求

```json
{
  "request_id": "eval:<run_id>:locomo_refined:conv-0:chunk-0",
  "messages": [{
    "role": "user",
    "timestamp": 1704067200000,
    "content": "raw memory text"
  }],
  "user_id": "eval:<run_id>:locomo:conv-0",
  "session_id": "eval:<run_id>:sample:0"
}
```

| 字段 | 要求 | 说明 |
|---|---|---|
| `request_id` | 必填 | 本次逻辑写入的标识；重试时保持不变，响应需原样回显。 |
| `messages` | 必填 | 按来源顺序排列的消息；请按收到的顺序处理。 |
| `messages[].role` | 必填 | 消息角色为 user 或 assistant。 |
| `messages[].content` | 必填 | 字符串。 |
| `messages[].timestamp` | 可选 | 源数据有时间戳时发送，单位为 Unix 毫秒；分段不改变其值和消息顺序。 |
| `user_id` | 必填 | 记忆隔离范围；后续 Search 使用相同的值。 |
| `session_id` | 必填 | 来源对话或会话的标识。 |

## Add 响应

```json
{
  "success": true,
  "request_id": "eval:<run_id>:locomo_refined:conv-0:chunk-0",
  "user_id": "eval:<run_id>:locomo:conv-0",
  "session_id": "eval:<run_id>:sample:0"
}
```

| 字段 | 要求 | 说明 |
|---|---|---|
| `success` | 必填 | 消息已持久化且可立即检索后，返回值必须为 true。 |
| `request_id` | 必填 | 原样返回 Add 请求中的 request_id。 |
| `user_id` | 必填 | 原样返回 Add 请求中的 user_id。 |
| `session_id` | 必填 | 原样返回 Add 请求中的 session_id。 |

## Search 请求

所有赛道的 options 都保持为顶层数组。

```json
{
  "query": "Which answer best matches the memory?",
  "options": ["A. First answer", "B. Second answer"],
  "user_id": "eval:<run_id>:locomo:conv-0",
  "top_k": 100
}
```

| 字段 | 要求 | 说明 |
|---|---|---|
| `query` | 必填 | 基准原题，字符串。 |
| `options` | 按题型发送 | 选择题（含 Streaming）在顶层发送选项；开放题不发送，且选项不含金标答案。 |
| `user_id` | 必填 | 与对应 Add 请求相同的记忆隔离范围。 |
| `top_k` | 必填 | 允许返回的最大记忆条数；正式外部评测固定为 100。 |

## Search 响应

请按检索排名顺序返回记忆。

```json
{
  "data": [{
    "id": "mem_123",
    "content": "remembered fact text",
    "score": 0.87,
    "created_at": "2026-07-01T12:00:00Z"
  }]
}
```

| 字段 | 要求 | 说明 |
|---|---|---|
| `data` | 必填 | 按检索排名排列的数组；无结果时返回 `[]`，不得省略 data。 |
| `data[].id` | 必填 | 每条返回记忆的稳定标识。 |
| `data[].content` | 必填 | 返回非空字符串；内容会原样保留供审计，并按返回顺序进入 Answer。 |
| `data[].score` | 可选 | 数值型相关性得分；数值越高必须表示相关性越强。 |
| `data[].created_at` | 可选 | 该记忆的来源时间或持久化时间。 |
