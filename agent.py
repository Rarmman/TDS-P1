"""
Single-file agentic Telegram bot.

Architecture (per spec):
  FastAPI web app  -> GET /health          (keep-alive + sanity check)
                   -> GET /run.jsonl       (the public agent log)
  Background thread -> Telegram getUpdates long-poll loop
                     -> per-message: agent loop -> sendMessage(JSON)
  Background thread -> self-ping /health every 10 min (free hosts idle out)

Run:
  pip install fastapi uvicorn requests pandas numpy openpyxl beautifulsoup4
  export TELEGRAM_BOT_TOKEN=...
  export LLM_API_KEY=...              # Gemini API key from aistudio.google.com
  export LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
  export PUBLIC_URL=https://your-app.onrender.com   # used for self-ping + log_url
  python agent.py
"""

import contextlib
import io
import json
import os
import re
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)

# CHANGED: single LLM_MODEL replaced with an ordered fallback chain.
# Ordered by actual RPD headroom confirmed on the AI Studio Rate Limit
# dashboard for this project -- NOT by version number. Re-check that
# dashboard periodically; these numbers can change.
#   gemini-3.5-flash-lite : 15 RPM / 500 RPD  (best headroom)
#   gemini-3.1-flash-lite : 15 RPM / 500 RPD  (near-identical backup)
#   gemini-2.5-flash-lite : 10 RPM /  20 RPD  (thin -- use sparingly)
#   gemini-2.5-flash      :  5 RPM /  20 RPD  (thinnest -- last resort)
MODEL_FALLBACK_CHAIN = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

PORT = int(os.environ.get("PORT", "8000"))
# Render sets RENDER_EXTERNAL_URL automatically; fall back to it if PUBLIC_URL isn't set.
PUBLIC_URL = os.environ.get("PUBLIC_URL") or os.environ.get(
    "RENDER_EXTERNAL_URL", f"http://localhost:{PORT}"
)

LOG_PATH = os.environ.get("LOG_PATH", "run.jsonl")
LOG_URL = f"{PUBLIC_URL.rstrip('/')}/run.jsonl"

MAX_TOOL_OUTPUT_CHARS = 8000
MAX_AGENT_STEPS = 10
WALL_CLOCK_BUDGET_SECONDS = 210
HISTORY_TURNS_PER_CHAT = 20
SELF_PING_INTERVAL_SECONDS = 10 * 60  # 10 min

# --------------------------------------------------------------------------
# Logging (JSONL) -- doubles as public log and debugging tool
# --------------------------------------------------------------------------

_log_lock = threading.Lock()


def log_event(event_type: str, chat_id=None, **fields):
    """Append one JSON line to LOG_PATH. Never raises."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "chat_id": chat_id,
        **fields,
    }
    try:
        with _log_lock:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Logging must never crash the handler.
        traceback.print_exc()


# --------------------------------------------------------------------------
# Per-chat history (last ~20 turns so multi-turn questions have context)
# --------------------------------------------------------------------------

_histories: dict[int, deque] = {}
_histories_lock = threading.Lock()


def get_history(chat_id: int) -> list:
    with _histories_lock:
        if chat_id not in _histories:
            _histories[chat_id] = deque(maxlen=HISTORY_TURNS_PER_CHAT)
        return list(_histories[chat_id])


def push_history(chat_id: int, role: str, content: str):
    with _histories_lock:
        if chat_id not in _histories:
            _histories[chat_id] = deque(maxlen=HISTORY_TURNS_PER_CHAT)
        _histories[chat_id].append({"role": role, "content": content})


# --------------------------------------------------------------------------
# The one tool: run_python
# --------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code server-side and return captured stdout. "
                "Use this to fetch/compute anything instead of guessing a number. "
                "pandas, numpy, requests, BeautifulSoup, and openpyxl are available, "
                "so you can download and analyse public datasets "
                "(e.g. MOSPI-published XLSX/CSV/HTML tables)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source to exec(). Print anything you need returned.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]


def run_python(code: str) -> str:
    """exec() the code, capture stdout, cap the output (last N chars)."""
    stdout_buf = io.StringIO()
    exec_globals = {"__name__": "__agent_sandbox__"}
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, exec_globals)
        output = stdout_buf.getvalue()
    except Exception:
        output = stdout_buf.getvalue() + "\n" + traceback.format_exc()

    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[-MAX_TOOL_OUTPUT_CHARS:]
    return output


# --------------------------------------------------------------------------
# JSON extraction (defensive layer #1)
# --------------------------------------------------------------------------

def _find_first_balanced_object(text: str) -> str | None:
    """Return the substring of the first balanced {...} block, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        # unbalanced from this start; try the next '{'
        start = text.find("{", start + 1)
    return None


def extract_json_answer(raw_text: str, real_log_url: str) -> dict:
    """
    Strip code fences, find the first balanced {...}, json.loads it.
    If there's no "answer" key, wrap it. Always overwrite log_url.
    """
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    candidate = _find_first_balanced_object(text)
    parsed = None
    if candidate is not None:
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = None

    if parsed is None:
        parsed = {"answer": text}
    elif "answer" not in parsed:
        parsed = {"answer": parsed}

    parsed["log_url"] = real_log_url
    return parsed


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are an autonomous agent answering exam-style questions sent over Telegram.

Rules you must follow:
1. Answer the LATEST message; earlier messages are context (multi-turn conversation).
2. Use the run_python tool to fetch or compute anything factual or numeric -- never
   guess a number you could compute. For published statistics where fetching fails,
   you may answer from your own knowledge.
3. Output ONLY the JSON object the question asks for -- no prose, no markdown code
   fences. Put a placeholder string in the "log_url" field; the calling code will
   substitute the real URL.
4. Match the requested "answer" shape exactly. Never add extra keys beyond what's
   asked for (aside from log_url).
5. If a message is only setup for a later message (e.g. "I'll send data next"),
   still reply with a small JSON ack such as {{"answer": "ok", "log_url": "placeholder"}}
   -- every message must get a reply.

Current log URL to use as the placeholder value: {LOG_URL}
"""


# --------------------------------------------------------------------------
# LLM call (OpenAI-compatible chat completions with function calling)
# CHANGED: now tries each model in MODEL_FALLBACK_CHAIN in order, moving to
# the next on any error (rate limit, 404, 5xx, timeout). Logs which model
# actually served each request so you can check /run.jsonl after grading to
# see if/when fallbacks fired.
# --------------------------------------------------------------------------

def call_llm(messages: list, tools_enabled: bool) -> dict:
    last_error = None

    for model_name in MODEL_FALLBACK_CHAIN:
        payload = {
            "model": model_name,
            "messages": messages,
        }
        if tools_enabled:
            payload["tools"] = TOOLS
            payload["tool_choice"] = "auto"

        try:
            resp = requests.post(
                f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            if not resp.ok:
                # CHANGED: capture the actual response body, not just the
                # status line -- this is what tells us WHY it's a 400.
                raise RuntimeError(
                    f"{resp.status_code} error from {model_name}: {resp.text[:1000]}"
                )
            data = resp.json()

            message = data["choices"][0]["message"]
            message["_model_used"] = model_name  # tag for logging in agent_loop
            return message

        except Exception as e:
            last_error = e
            log_event("llm_call_failed", model=model_name, error=str(e)[-1000:])
            continue  # try next model in the chain

    # every model in the chain failed
    raise RuntimeError(f"All models in fallback chain failed. Last error: {last_error}")


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------

def agent_loop(chat_id: int, user_message: str) -> dict:
    """
    Loop: send conversation -> if the model calls the tool, run it, append
    result, repeat (cap at ~MAX_AGENT_STEPS) -> when the model answers with
    plain text, parse the JSON out of it.
    """
    start_time = time.time()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(get_history(chat_id))
    messages.append({"role": "user", "content": user_message})

    push_history(chat_id, "user", user_message)

    final_text = None
    model_used_for_final = None  # CHANGED: track which model produced the final answer

    for step in range(MAX_AGENT_STEPS):
        elapsed = time.time() - start_time
        past_budget = elapsed > WALL_CLOCK_BUDGET_SECONDS
        tools_enabled = not past_budget

        if past_budget:
            # Force an immediate answer, no more tool calls.
            messages.append(
                {
                    "role": "user",
                    "content": "Time budget exceeded. Answer NOW with the JSON object only, no more tool calls.",
                }
            )

        message = call_llm(messages, tools_enabled=tools_enabled)
        model_used_for_final = message.get("_model_used")
        tool_calls = message.get("tool_calls")

        if tool_calls and tools_enabled:
            messages.append(message)
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    args = {}

                if fn_name == "run_python":
                    code = args.get("code", "")
                    log_event("tool_call", chat_id=chat_id, step=step, tool=fn_name, code=code)
                    result = run_python(code)
                    log_event("tool_result", chat_id=chat_id, step=step, tool=fn_name, output=result)
                else:
                    result = f"Unknown tool: {fn_name}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )
            continue  # go around the loop again with tool results appended

        # No tool call (or budget forced a stop): this is the final answer.
        final_text = message.get("content") or ""
        break

    if final_text is None:
        final_text = '{"answer": "no response produced"}'

    answer_json = extract_json_answer(final_text, LOG_URL)
    push_history(chat_id, "assistant", json.dumps(answer_json))

    # CHANGED: record which model actually produced this answer, for debugging
    log_event("model_used", chat_id=chat_id, model=model_used_for_final)

    return answer_json


def handle_incoming_message(chat_id: int, question: str) -> dict:
    """Never crash silently: wrap the handler in try/except."""
    log_event("question", chat_id=chat_id, question=question)
    try:
        answer_json = agent_loop(chat_id, question)
    except Exception:
        # FIX (bug 1): don't put the traceback in the reply itself -- it's an
        # extra key that breaks exact-match grading. Log it instead.
        log_event("internal_error", chat_id=chat_id, error=traceback.format_exc()[-2000:])
        answer_json = {"answer": "internal error", "log_url": LOG_URL}
    log_event("final_reply", chat_id=chat_id, reply=answer_json)
    return answer_json


# --------------------------------------------------------------------------
# Telegram plumbing
# --------------------------------------------------------------------------

def telegram_send_message(chat_id: int, text: str):
    try:
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=30,
        )
    except Exception:
        traceback.print_exc()


def telegram_poll_loop():
    """Background thread: Telegram getUpdates long-poll loop."""
    offset = None
    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset

            resp = requests.get(f"{TELEGRAM_API}/getUpdates", params=params, timeout=40)
            resp.raise_for_status()
            data = resp.json()

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message") or update.get("edited_message")
                if not message:
                    continue

                chat_id = message["chat"]["id"]

                # FIX (bug 2): don't silently drop non-text messages -- the
                # grader must get a reply to every message, including files.
                text = message.get("text") or message.get("caption") or ""
                if not text and "document" in message:
                    text = f"[User sent a file: {message['document'].get('file_name', 'unknown')}]"
                if not text:
                    continue

                answer_json = handle_incoming_message(chat_id, text)
                telegram_send_message(chat_id, json.dumps(answer_json))

        except Exception:
            traceback.print_exc()
            time.sleep(5)


def self_ping_loop():
    """Background thread: self-ping /health every 10 min (free hosts idle out)."""
    health_url = f"{PUBLIC_URL.rstrip('/')}/health"
    while True:
        time.sleep(SELF_PING_INTERVAL_SECONDS)
        try:
            requests.get(health_url, timeout=15)
        except Exception:
            traceback.print_exc()


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/run.jsonl")
def get_run_log():
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""
    return PlainTextResponse(content, media_type="application/x-ndjson")


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------

def main():
    threading.Thread(target=telegram_poll_loop, daemon=True).start()
    threading.Thread(target=self_ping_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
