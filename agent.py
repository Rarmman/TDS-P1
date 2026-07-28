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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
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

# Soft budget: once elapsed time inside agent_loop passes this, stop issuing
# new tool calls and ask the model to answer immediately. This shapes
# behavior but does NOT guarantee an upper bound -- the forced "final answer"
# call can itself still be slow (see call_llm's per-model timeout * fallback
# chain length), and a hung tool call isn't bounded by this at all.
WALL_CLOCK_BUDGET_SECONDS = 210

# Hard ceiling: handle_incoming_message will ALWAYS reply within this many
# seconds of receiving a message, no matter what happens inside agent_loop
# (hung LLM call, hung tool exec, every fallback model timing out, etc.).
# Keep this comfortably above WALL_CLOCK_BUDGET_SECONDS so the soft-budget
# forced-answer path normally has time to finish and be used, and adjust to
# whatever your grader's actual per-message timeout is (with a safety margin
# below it, since Telegram delivery + grader-side overhead also eat time).
HARD_REPLY_DEADLINE_SECONDS = int(os.environ.get("HARD_REPLY_DEADLINE_SECONDS", "240"))

HISTORY_TURNS_PER_CHAT = 20
SELF_PING_INTERVAL_SECONDS = 10 * 60  # 10 min

# Worker pool that runs agent_loop() so handle_incoming_message can enforce
# HARD_REPLY_DEADLINE_SECONDS from the outside via future.result(timeout=...).
# max_workers caps how many questions can be "in flight" (mid-agent-loop,
# including ones we've already given up waiting on) at once; daemon-like
# behavior isn't needed here since these are plain worker threads, not the
# long-lived polling threads.
_agent_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agent-loop")


def _mask_key(key: str) -> str:
    """Return a safe-to-log fingerprint of a secret: length + first/last 4 chars only."""
    if not key:
        return "EMPTY"
    if len(key) <= 8:
        return f"len={len(key)} (too short to fingerprint safely)"
    return f"len={len(key)} starts={key[:4]}... ends=...{key[-4:]}"

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


RUN_PYTHON_TIMEOUT_SECONDS = 60  # cap on a single run_python call


def run_python(code: str) -> str:
    """
    exec() the code in a daemon thread, capped at RUN_PYTHON_TIMEOUT_SECONDS.
    Python can't forcibly kill a thread, so a still-hanging thread is simply
    abandoned (daemon=True means it won't block process shutdown) -- but this
    call always returns promptly, which is what keeps a stuck run_python from
    silently eating the entire HARD_REPLY_DEADLINE_SECONDS budget.
    """
    stdout_buf = io.StringIO()
    done = {}

    def _target():
        exec_globals = {"__name__": "__agent_sandbox__"}
        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, exec_globals)
            done["ok"] = True
        except Exception:
            done["ok"] = False
            stdout_buf.write("\n" + traceback.format_exc())

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(RUN_PYTHON_TIMEOUT_SECONDS)

    if t.is_alive():
        output = stdout_buf.getvalue() + f"\nERROR: code timed out after {RUN_PYTHON_TIMEOUT_SECONDS}s"
    else:
        output = stdout_buf.getvalue()

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
    """
    Never crash silently, and never take longer than HARD_REPLY_DEADLINE_SECONDS
    to produce a reply -- no matter what happens inside agent_loop (a hung
    LLM call, a hung tool call, all fallback models being slow, etc.).

    We run agent_loop in a worker thread and bound it with future.result(timeout=...).
    WALL_CLOCK_BUDGET_SECONDS (inside agent_loop) shapes *behavior* -- it tries
    to get a clean answer out cooperatively. HARD_REPLY_DEADLINE_SECONDS here is
    the actual guarantee: if agent_loop doesn't return in time for ANY reason,
    we reply immediately with a fallback JSON instead of leaving the grader
    waiting. The abandoned thread is left to finish on its own (it may still
    log tool results / errors), but it can never send a second Telegram
    message -- only telegram_poll_loop does that, using our returned value.
    """
    log_event("question", chat_id=chat_id, question=question)

    future = _agent_executor.submit(agent_loop, chat_id, question)
    try:
        answer_json = future.result(timeout=HARD_REPLY_DEADLINE_SECONDS)
    except FutureTimeoutError:
        log_event(
            "hard_deadline_exceeded",
            chat_id=chat_id,
            budget_seconds=HARD_REPLY_DEADLINE_SECONDS,
        )
        answer_json = {"answer": "timed out", "log_url": LOG_URL}
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


# Pool that update-processing runs on, so handling one message (which can
# legitimately take up to HARD_REPLY_DEADLINE_SECONDS) never blocks
# telegram_poll_loop from calling getUpdates again for everyone else.
_update_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="update-handler")

# Per-chat FIFO ordering: without this, two messages from the same chat could
# both land on the pool above and run concurrently, so a slow reply to an
# earlier message could arrive AFTER the reply to a later one. To guarantee
# in-order replies within a chat (while still letting different chats run in
# parallel), each chat gets its own queue and at most one active worker;
# _chat_queue_lock protects both structures together so a queued message can
# never be "orphaned" between a worker finishing and a new message arriving.
_chat_queues: dict[int, list] = {}
_chat_active: set = set()
_chat_queue_lock = threading.Lock()


def _enqueue_update(chat_id: int, text: str):
    with _chat_queue_lock:
        _chat_queues.setdefault(chat_id, []).append(text)
        if chat_id in _chat_active:
            # A worker is already draining this chat's queue -- it will pick
            # up this message when it gets to it. Don't start a second one.
            return
        _chat_active.add(chat_id)
    _update_executor.submit(_drain_chat_queue, chat_id)


def _drain_chat_queue(chat_id: int):
    """Process this chat's queued messages one at a time, oldest first."""
    while True:
        with _chat_queue_lock:
            queue = _chat_queues.get(chat_id) or []
            if not queue:
                _chat_active.discard(chat_id)
                return
            text = queue.pop(0)
        _process_update(chat_id, text)


def _process_update(chat_id: int, text: str):
    print(f"[telegram_poll_loop] processing message from chat {chat_id}: {text[:80]!r}", flush=True)
    try:
        answer_json = handle_incoming_message(chat_id, text)
        telegram_send_message(chat_id, json.dumps(answer_json))
    except Exception:
        print(f"[_process_update] EXCEPTION for chat {chat_id}:", flush=True)
        traceback.print_exc()


def telegram_poll_loop():
    """Background thread: Telegram getUpdates long-poll loop."""
    print(f"[telegram_poll_loop] starting, bot token prefix: {TELEGRAM_BOT_TOKEN[:8]}...", flush=True)
    offset = None
    loop_count = 0
    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset

            resp = requests.get(f"{TELEGRAM_API}/getUpdates", params=params, timeout=40)
            resp.raise_for_status()
            data = resp.json()

            loop_count += 1
            if loop_count % 5 == 0:
                # CHANGED: periodic heartbeat so Render logs show this loop is alive,
                # not just silent until something goes wrong.
                print(f"[telegram_poll_loop] heartbeat #{loop_count}, ok={data.get('ok')}", flush=True)

            if not data.get("ok"):
                # CHANGED: surface Telegram API-level errors explicitly (e.g. 409
                # Conflict from a webhook being set, or an invalid token) instead
                # of silently doing nothing.
                print(f"[telegram_poll_loop] Telegram API returned not-ok: {data}", flush=True)

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

                # CHANGED: enqueue per-chat instead of submitting straight to
                # the pool -- guarantees in-order replies within a chat while
                # still processing different chats concurrently. offset is
                # already advanced above, so this can never cause Telegram to
                # redeliver the same update even if it's still queued/processing.
                _enqueue_update(chat_id, text)

        except Exception as e:
            print(f"[telegram_poll_loop] EXCEPTION: {e}", flush=True)
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

# CRITICAL FIX (see history): Render's start command is `uvicorn agent:app ...`,
# which imports this file and grabs `app` -- it never executes an
# `if __name__ == "__main__":` block, so threads started only there never run
# on Render at all.
#
# Using @app.on_event("startup") (rather than starting threads at bare module
# import time) ties thread startup to FastAPI's actual app lifecycle: it's
# guaranteed to fire exactly once, right when uvicorn begins serving this
# app, regardless of whether that's `uvicorn agent:app ...` from the CLI or
# `uvicorn.run(app)` from main() below.
@app.on_event("startup")
def _start_background_threads():
    threading.Thread(target=telegram_poll_loop, daemon=True).start()
    threading.Thread(target=self_ping_loop, daemon=True).start()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ts": datetime.now(timezone.utc).isoformat(),
        # TEMPORARY DEBUG FIELDS -- remove before final submission.
        # Never logs/exposes the actual key, only length + first/last 4 chars.
        "llm_api_key_fingerprint": _mask_key(LLM_API_KEY),
        "llm_base_url": LLM_BASE_URL,
        "model_fallback_chain": MODEL_FALLBACK_CHAIN,
    }


@app.get("/debug-llm")
def debug_llm():
    """
    TEMPORARY DEBUG ENDPOINT -- remove before final submission.
    Makes one raw test call to the first model in the fallback chain,
    from inside the running Render container, and returns exactly what
    Google's API said. This tests auth/network from Render's own network
    without needing shell access.
    """
    model_name = MODEL_FALLBACK_CHAIN[0]
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
    }
    try:
        resp = requests.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        return {
            "model_tried": model_name,
            "status_code": resp.status_code,
            "response_body": resp.text[:2000],
            "key_fingerprint": _mask_key(LLM_API_KEY),
        }
    except Exception as e:
        return {
            "model_tried": model_name,
            "error": str(e)[:2000],
            "key_fingerprint": _mask_key(LLM_API_KEY),
        }


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
    # NOTE: telegram_poll_loop and self_ping_loop are started by the
    # @app.on_event("startup") handler above, which fires automatically once
    # uvicorn.run(app, ...) below actually starts serving -- so they run
    # correctly whether this is invoked via `python agent.py` (uvicorn.run
    # here) or via Render's `uvicorn agent:app ...` (uvicorn's own CLI).
    # Do NOT start them again here, or they'd run twice.
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
