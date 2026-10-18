"""DeepSeek Chat Completions 适配器（T15）。

职责单一：把「事实摘要 + 问题」交给 DeepSeek，把回答和真实状态原样带回来。
不做流程编排、不碰数据库、不改任何治理状态。

三条不可放松的约束：

1. 密钥只从环境变量 ``DEEPSEEK_API_KEY`` 或项目根 ``.env`` 读取。
   不写进代码、不写进 SQLite、不进日志、不进报告、不上前端。
   任何对外文本都先过 :func:`redact`。
2. 失败就如实报失败：``answer`` 留空并给出明确状态码，
   绝不返回编造的答案，也不把异常包装成"看起来像回答"的文本。
3. 重试有明确上限：默认最多 2 次（首次 + 一次重试），且只对
   超时 / 限流 / 5xx 这类瞬时故障重试；认证失败、余额不足、参数错误一律不重试。
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from . import config as _config

# ── 环境变量名 ────────────────────────────────────────────────────────────
ENV_KEY = "DEEPSEEK_API_KEY"
ENV_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_MODEL = "DEEPSEEK_MODEL"
ENV_TIMEOUT = "DEEPSEEK_TIMEOUT"
ENV_MAX_TOKENS = "DEEPSEEK_MAX_TOKENS"
ENV_TEMPERATURE = "DEEPSEEK_TEMPERATURE"
ENV_MAX_ATTEMPTS = "DEEPSEEK_MAX_ATTEMPTS"

# ── 默认值（模型名与参数都可配置，见 docs/DeepSeek问答.md）───────────────
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_TOKENS = 900
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_ATTEMPTS = 2
# 硬上限：无论环境变量写多大都不超过这里，避免出现无上限重试。
HARD_MAX_ATTEMPTS = 3

ENV_FILE = _config.PROJECT_ROOT / ".env"

# ── 状态码与中文说明 ──────────────────────────────────────────────────────
STATUS_ZH: dict[str, str] = {
    "ok": "成功",
    "not_configured": "未配置 API Key",
    "auth_failed": "认证失败（Key 无效）",
    "insufficient_balance": "账户余额不足",
    "rate_limited": "触发限流",
    "invalid_request": "请求参数无效",
    "server_error": "服务端错误",
    "timeout": "请求超时",
    "network_error": "网络不可达",
    "bad_response": "返回内容不可解析",
}

# 这些状态属于瞬时故障，允许按上限重试一次。
RETRYABLE = {"timeout", "rate_limited", "server_error", "network_error"}

# 这些状态说明"这次根本没发出有效请求"，页面据此提示本地查询仍可用。
NO_REQUEST_SENT = {"not_configured"}

SYSTEM_PROMPT = (
    "你是农业数据治理系统的问答助手。用户会给你一份【事实摘要】和一个【问题】。\n"
    "必须遵守：\n"
    "1. 只使用【事实摘要】里出现的数据回答。摘要里没有的数字、场景、文件，一律不得编造。\n"
    "2. 摘要中标注为「缺口」的内容，要如实说明缺什么，不要用常识补齐。\n"
    "3. 统计口径以摘要为准；摘要已经算好的百分比和计数直接引用，不要自己重新计算。\n"
    "4. 你只做解释和建议，不能修改任何数据、质量状态或执行 SQL；如需处置，只能建议人去操作。\n"
    "5. 回答用简体中文，先给结论再用短句分点说明，不超过 400 字。\n"
    "6. 若摘要里没有与问题相关的数据，直接说「当前资料不足以回答这个问题」，并指出缺哪类资料。"
)

_ABS_PATH = re.compile(r"[A-Za-z]:\\[^\s\"'）)，,；;]*")
_SK_LIKE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")


# ── 密钥脱敏 ──────────────────────────────────────────────────────────────
def redact(text: str, secret: str = "") -> str:
    """把密钥从任意文本里抹掉，供日志/异常/报告统一使用。

    除了整串替换，还要处理接口的一种回显：认证失败时服务端返回
    ``Your api key: ****0000 is invalid``，即只露后四位。
    这里连同 ``****+后四位`` 一起抹掉，避免错误信息把尾巴带出去。
    """
    if not text:
        return ""
    if secret and len(secret) >= 8:
        text = text.replace(secret, "sk-***")
        text = re.sub(r"\*{2,}" + re.escape(secret[-4:]), "****", text)
    return _SK_LIKE.sub("sk-***", text)


def sanitize_facts_text(text: str) -> str:
    """去掉事实摘要里的本地绝对路径，只把业务事实发给模型。

    质检 reason 原文里带有 ``D:\\...\\store\\objects\\<sha256>`` 这类本机路径，
    它们对回答没有用处，属于不该外发的本地信息。
    """
    if not text:
        return ""
    cleaned = _ABS_PATH.sub("<本机路径已省略>", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


# ── 配置 ──────────────────────────────────────────────────────────────────
def read_env_file(path: Path | None = None) -> dict[str, str]:
    """读取项目根 ``.env``；文件不存在或不可读时返回空字典，不抛异常。

    只做最小解析，避免为一个键值文件引入额外依赖。
    支持 ``export KEY=VALUE``、单双引号包裹和 ``#`` 整行注释。
    """
    target = Path(path) if path is not None else ENV_FILE
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def os_env(name: str) -> str:
    """读环境变量并去掉首尾空白；未设置时返回空串。"""
    return os.environ.get(name, "").strip()


def _pick(name: str, env_file: dict[str, str], default: str) -> str:
    """环境变量优先，其次 .env，最后默认值。"""
    return (os_env(name) or env_file.get(name) or default).strip()


def _as_float(raw: str, default: float, low: float, high: float) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    return min(max(v, low), high)


def _as_int(raw: str, default: int, low: int, high: int) -> int:
    try:
        v = int(float(raw))
    except (TypeError, ValueError):
        return default
    return min(max(v, low), high)


@dataclass(frozen=True)
class DeepSeekConfig:
    """一次调用所需的全部配置。``describe()`` 不会带出密钥。"""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    key_source: str = "未找到"          # 环境变量 / .env / 未找到
    env_file: str = str(ENV_FILE)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def key_hint(self) -> str:
        """只暴露"有没有配、配在哪"，不暴露密钥本身。"""
        if not self.api_key:
            return "未配置"
        return f"已配置（{self.key_source}，长度 {len(self.api_key)}）"

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_attempts": self.max_attempts,
            "configured": self.configured,
            "key_hint": self.key_hint,
            "key_source": self.key_source,
            "env_file": self.env_file,
        }

    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


def load_config(env_file: dict[str, str] | None = None) -> DeepSeekConfig:
    """按「环境变量 → .env → 默认值」的优先级组装配置。"""
    table = read_env_file() if env_file is None else env_file
    key_env = os_env(ENV_KEY)
    key_file = table.get(ENV_KEY, "")
    api_key = (key_env or key_file).strip()
    if key_env:
        source = "环境变量"
    elif key_file:
        source = ".env 文件"
    else:
        source = "未找到"
    return DeepSeekConfig(
        api_key=api_key,
        base_url=_pick(ENV_BASE_URL, table, DEFAULT_BASE_URL),
        model=_pick(ENV_MODEL, table, DEFAULT_MODEL),
        timeout=_as_float(_pick(ENV_TIMEOUT, table, str(DEFAULT_TIMEOUT)), DEFAULT_TIMEOUT, 1.0, 300.0),
        max_tokens=_as_int(_pick(ENV_MAX_TOKENS, table, str(DEFAULT_MAX_TOKENS)), DEFAULT_MAX_TOKENS, 16, 8192),
        temperature=_as_float(
            _pick(ENV_TEMPERATURE, table, str(DEFAULT_TEMPERATURE)), DEFAULT_TEMPERATURE, 0.0, 2.0
        ),
        max_attempts=_as_int(
            _pick(ENV_MAX_ATTEMPTS, table, str(DEFAULT_MAX_ATTEMPTS)),
            DEFAULT_MAX_ATTEMPTS, 1, HARD_MAX_ATTEMPTS,
        ),
        key_source=source,
        env_file=str(ENV_FILE),
    )


# ── 返回结构 ──────────────────────────────────────────────────────────────
@dataclass
class DeepSeekReply:
    """一次问答的真实结果。失败时 ``answer`` 必为空字符串。"""

    ok: bool = False
    status: str = "not_configured"
    status_zh: str = STATUS_ZH["not_configured"]
    answer: str = ""
    model: str = ""            # 本次**请求**的模型（来自配置）
    response_model: str = ""   # 服务端**回报**的模型，可能因别名解析而与请求值不同
    http_status: int | None = None
    error_code: str = ""
    error_message: str = ""
    attempts: int = 0
    latency_ms: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    provider_request_id: str = ""
    retried: bool = False

    def as_record(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "status_zh": self.status_zh,
            "answer": self.answer,
            "model": self.model,
            "response_model": self.response_model,
            "http_status": self.http_status,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "attempts": self.attempts,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.usage.get("prompt_tokens"),
            "completion_tokens": self.usage.get("completion_tokens"),
            "total_tokens": self.usage.get("total_tokens"),
            "provider_request_id": self.provider_request_id,
            "retried": self.retried,
        }


def _classify_http(status_code: int) -> str:
    """把 DeepSeek 文档里的 HTTP 状态码映射成稳定的内部状态。"""
    if status_code == 401:
        return "auth_failed"
    if status_code == 402:
        return "insufficient_balance"
    if status_code == 429:
        return "rate_limited"
    if status_code in (400, 422):
        return "invalid_request"
    if status_code in (500, 502, 503, 504):
        return "server_error"
    if 400 <= status_code < 500:
        return "invalid_request"
    return "server_error"


def _extract_error(payload: Any) -> tuple[str, str]:
    """从 error 响应里取 (code, message)，取不到就给可读兜底。"""
    if not isinstance(payload, dict):
        return "", ""
    err = payload.get("error")
    if not isinstance(err, dict):
        return "", ""
    code = str(err.get("code") or err.get("type") or "")
    message = str(err.get("message") or "")
    return code, message


def chat_ready(config: DeepSeekConfig | None = None) -> tuple[bool, str]:
    """不联网，只判断"能不能发起调用"，供页面提前给出提示。"""
    cfg = config or load_config()
    if not cfg.configured:
        return False, (
            f"未配置 {ENV_KEY}。请在项目根 .env 文件或环境变量中填写后再提问；"
            "本地事实摘要与数据库查询不受影响。"
        )
    return True, f"已配置（{cfg.key_source}）；模型 {cfg.model}，地址 {cfg.base_url}"


# ── 主入口 ────────────────────────────────────────────────────────────────
def ask(
    question: str,
    facts_text: str,
    *,
    config: DeepSeekConfig | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    session: requests.Session | None = None,
) -> DeepSeekReply:
    """把事实摘要与问题发给 DeepSeek，返回带状态的结果。

    不抛异常：所有失败都落在 :class:`DeepSeekReply` 的 status 上，
    调用方据此显示准确状态，而不是拿到一个假的回答。
    """
    cfg = config or load_config()
    question = (question or "").strip()
    reply = DeepSeekReply(model=cfg.model)

    if not cfg.configured:
        reply.status = "not_configured"
        reply.status_zh = STATUS_ZH["not_configured"]
        reply.error_message = f"缺少 {ENV_KEY}，未发起请求。"
        return reply

    if not question:
        reply.status = "invalid_request"
        reply.status_zh = STATUS_ZH["invalid_request"]
        reply.error_message = "问题为空，未发起请求。"
        return reply

    facts = sanitize_facts_text(facts_text or "")
    user_content = f"【事实摘要】\n{facts}\n\n【问题】\n{question}"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
    }

    own_session = session is None
    http = session or requests.Session()
    started = time.monotonic()
    try:
        for attempt in range(1, cfg.max_attempts + 1):
            reply.attempts = attempt
            reply.retried = attempt > 1
            try:
                resp = http.post(
                    cfg.endpoint(), headers=headers, json=payload, timeout=cfg.timeout
                )
            except requests.Timeout:
                reply.status = "timeout"
                reply.status_zh = STATUS_ZH["timeout"]
                reply.error_message = f"请求超过 {cfg.timeout:g} 秒未返回。"
            except requests.RequestException as exc:
                reply.status = "network_error"
                reply.status_zh = STATUS_ZH["network_error"]
                reply.error_message = redact(f"无法连接 {cfg.base_url}：{exc}", cfg.api_key)
            else:
                reply.http_status = resp.status_code
                body = _safe_json(resp)
                if resp.status_code == 200:
                    return _finish_ok(reply, body, cfg, started)
                code, message = _extract_error(body)
                reply.status = _classify_http(resp.status_code)
                reply.status_zh = STATUS_ZH.get(reply.status, "未知错误")
                reply.error_code = redact(code, cfg.api_key)
                reply.error_message = redact(
                    message or _raw_snippet(resp), cfg.api_key
                )
                reply.latency_ms = int((time.monotonic() - started) * 1000)

            # 到这里说明本次尝试失败：只有瞬时故障且未达上限才继续
            transient = reply.status in RETRYABLE
            if not transient or attempt >= cfg.max_attempts:
                reply.latency_ms = int((time.monotonic() - started) * 1000)
                return reply
            time.sleep(min(2.0 * attempt, 4.0))
        reply.latency_ms = int((time.monotonic() - started) * 1000)
        return reply
    finally:
        if own_session:
            http.close()


def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def _raw_snippet(resp: requests.Response, limit: int = 200) -> str:
    try:
        return (resp.text or "")[:limit]
    except Exception:  # pragma: no cover - 仅在极端情况下触发
        return ""


def _finish_ok(
    reply: DeepSeekReply, body: Any, cfg: DeepSeekConfig, started: float
) -> DeepSeekReply:
    """解析 200 响应；结构不对就判 bad_response，不硬凑答案。"""
    reply.latency_ms = int((time.monotonic() - started) * 1000)
    if not isinstance(body, dict):
        reply.status = "bad_response"
        reply.status_zh = STATUS_ZH["bad_response"]
        reply.error_message = "响应不是合法 JSON 对象。"
        return reply
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        reply.status = "bad_response"
        reply.status_zh = STATUS_ZH["bad_response"]
        reply.error_message = "响应中没有 choices。"
        return reply
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = (message or {}).get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        reply.status = "bad_response"
        reply.status_zh = STATUS_ZH["bad_response"]
        reply.error_message = "响应中 message.content 为空。"
        return reply

    usage = body.get("usage") or {}
    reply.ok = True
    reply.status = "ok"
    reply.status_zh = STATUS_ZH["ok"]
    reply.answer = redact(content.strip(), cfg.api_key)
    # 请求值原样保留；服务端回报值单独记，避免"请求 deepseek-chat、
    # 实际由 deepseek-flash 承接"这类别名解析被抹掉。
    reply.model = cfg.model
    reply.response_model = str(body.get("model") or "")
    reply.provider_request_id = str(body.get("id") or "")
    reply.usage = {
        "prompt_tokens": _int_or_0(usage.get("prompt_tokens")),
        "completion_tokens": _int_or_0(usage.get("completion_tokens")),
        "total_tokens": _int_or_0(usage.get("total_tokens")),
    }
    reply.error_code = ""
    reply.error_message = ""
    return reply


def _int_or_0(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
