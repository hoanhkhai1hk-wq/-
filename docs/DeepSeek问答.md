# DeepSeek 数据问答：配置与使用说明（T15）

日期：2026-10-18。对应任务 `docs/开发任务清单.md` 的 T15。
接口依据：[官方 Chat Completions 文档](https://api-docs.deepseek.com/api/create-chat-completion/) 与 [错误码说明](https://api-docs.deepseek.com/quick_start/error_codes/)，接入当日已复核。

## 这是什么

网页第 9 页「DeepSeek 问答」。它把治理库里**已经存在的事实**整理成摘要，
连同问题一起交给 DeepSeek，由模型写解释；模型不参与任何计算，也不写库。

分工是硬的：

| 环节 | 谁做 | 产物 |
|---|---|---|
| 取事实 | `agrolake/qa_facts.py`，走 SQLite 只读连接 | 事实摘要文本 + 引用 + 结构化快照 |
| 写解释 | `agrolake/deepseek_client.py` 调 DeepSeek | 回答文本 / 明确状态 |
| 留档 | `agrolake/qa_store.py` 写 `qa_runs` | 问题、快照、回答、模型、状态、用量 |
| 展示 | `agrolake_web/qa_page.py` | 事实、回答、引用、历史 |

## 配置

三处，优先级从高到低：

1. **环境变量**（临时覆盖最方便）
2. **项目根 `.env`**（本机长期使用；已被 `.gitignore` 排除，不会进 Git）
3. **代码内默认值**（`agrolake/deepseek_client.py` 顶部常量）

| 变量 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | 是 | 空 | 未填时页面显示「未配置」，不发请求 |
| `DEEPSEEK_BASE_URL` | 否 | `https://api.deepseek.com` | 只填到域名，代码自己拼 `/chat/completions` |
| `DEEPSEEK_MODEL` | 否 | `deepseek-chat` | 换模型只改这一行，不用动业务代码 |
| `DEEPSEEK_TIMEOUT` | 否 | `30` | 秒，限 1-300 |
| `DEEPSEEK_MAX_TOKENS` | 否 | `900` | 单次回答上限，限 16-8192 |
| `DEEPSEEK_TEMPERATURE` | 否 | `0.2` | 事实问答建议保持低温 |
| `DEEPSEEK_MAX_ATTEMPTS` | 否 | `2` | 硬上限 3，超了按 3 处理 |

首次使用：

```powershell
Copy-Item .env.example .env
# 用记事本打开 .env，把 DEEPSEEK_API_KEY= 后面补上自己的 Key
```

## 页面怎么用

1. 打开 `streamlit run app.py`，切到第 9 个标签「DeepSeek 问答」。
2. 选「问题类型」，它决定取哪一份事实（页面会先显示这份事实，不用等模型）。
3. 输入问题，或直接用预填的示例问题，点「提问」。
4. 页面依次显示：事实摘要 → 回答（含状态、模型、耗时、token）→ 引用来源 → 历史记录。

**只有点「提问」才会发请求。** 切标签、改选项、普通 rerun 都不会触发调用，
不会在你没注意的时候反复计费。

三类可操作的问题（T15 成功验收要求）：

| 问题类型 | 示例问题 | 用的事实 |
|---|---|---|
| 当前批次待复核资料 | 当前批次待复核的资料有哪些？分别是什么原因？ | 质检阻断/警告、处置待确认、记录问题标记 |
| 场景质量与识别结果 | S1 这个场景的质量和识别结果怎么样？还缺哪些资料？ | 该场景成员版本、逐份质检、属性汇总、识别结果 |
| 治理报告 | 基于现在的实际指标，给出一份治理报告 | 概览指标 + 待复核清单 + 场景明细 |

第四类「治理概览」只讲统计口径，可作为追问的起点。

## 故障状态

失败一律显示准确状态，`answer` 留空，页面明确写「本次没有可用的模型回答，页面不会用模板话术冒充答案」。

| 状态 | 触发 | 页面显示 | 自动重试 |
|---|---|---|---|
| `not_configured` | 没配 Key | 未配置 API Key，给出配置指引 | 否（根本没发请求） |
| `auth_failed` | HTTP 401 | 认证失败（Key 无效） | 否 |
| `insufficient_balance` | HTTP 402 | 账户余额不足 | 否 |
| `invalid_request` | HTTP 400 / 422 | 请求参数无效 | 否 |
| `rate_limited` | HTTP 429 | 触发限流 | 是，最多一次 |
| `server_error` | HTTP 5xx | 服务端错误 | 是，最多一次 |
| `timeout` | 超过 `DEEPSEEK_TIMEOUT` | 请求超时 | 是，最多一次 |
| `network_error` | 连接失败 | 网络不可达 | 是，最多一次 |
| `bad_response` | 200 但内容为空/非 JSON | 返回内容不可解析 | 否 |

**认证、余额、参数错误不重试**——重试只会白花时间，不会变成功。
**上限是硬的**：`DEEPSEEK_MAX_ATTEMPTS` 写多大都按 3 封顶。

任何失败下，页面上的事实摘要和本地数据库查询都照常可用。

## 安全边界

- **密钥**只出现在 `Authorization` 请求头里。请求体、数据库、页面、日志、报告都不含它；
  `qa_runs` 表结构里没有任何能放密钥的列，写入前还会再过一遍 `redact()`。
- **只读**：事实查询用 `sqlite3` 的 `mode=ro` 连接，写操作在数据库层面就被拒绝，
  不靠"调用方自觉"。页面没有自由 SQL 入口，也没有改数据/改质量状态的按钮。
- **模型建议不落地**：模型只能给建议，处置仍需人到「问题处置」页操作。
- **不外发本机信息**：质检 reason 里的 `D:\...\store\objects\<sha256>` 会被替换成
  `<本机路径已省略>` 再发给模型。

## 一个实测发现

配置里写 `DEEPSEEK_MODEL=deepseek-chat`，接口返回 200 正常，
但响应体的 `model` 字段是 `deepseek-flash`——**服务端把 `deepseek-chat` 当别名解析到了 flash**。
代码因此分开记两个字段：`model`（请求值）与 `response_model`（服务端回报值），
页面上两者不同时会并列显示「请求 deepseek-chat（服务端实际 deepseek-flash）」，
不会偷偷把请求值改掉。

要换模型，改 `DEEPSEEK_MODEL` 即可；`deepseek-flash` 同样实测可用。

## 验证入口

```powershell
# 逻辑测试（36 项，不联网）
.\.venv-p0\Scripts\python.exe tests\p0\test_deepseek_qa.py

# 页面测试（4 项，AppTest，不联网）
.\.venv-p0\Scripts\python.exe tests\web\test_qa_page.py
```

真实调用记录见 `reports/DeepSeek接入验收.md` 与 `reports/DeepSeek接入验收-原始记录.json`。

## 已知限制

- 事实查询覆盖实际已存记录。FTW已有公开适配样例运行，不能当独立农业评估；
  RGB任务另有专用只读事实范围，可选择任务查看像素统计、候选区域与P0派生资产登记。
  缺库、未知任务、产物损坏或SQL明细不符时显示准确缺口，不补造成功统计。
- 回答质量取决于事实摘要的完整度；摘要里没有的内容，模型被要求直接说"资料不足"。
- 没有做多轮对话：每次提问都是独立的一次事实查询 + 一次回答，不做上下文累积。
- 未做并发/限流保护：单用户本地使用场景，多人同时提问可能触发 429，此时会如实显示。

RGB任务与登记步骤见[RGB治理与问答](RGB治理与问答.md)。回答绑定问题范围、实际任务及事实内容；生成时间变化不会使回答消失，任务或事实改变后旧回答不会挂在新引用下。2026-10-21这轮新增范围只做离线桩问答与真实浏览器事实显示验证，没有新增DeepSeek在线请求。
