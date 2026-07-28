# Data-Analyst Telegram Bot

Single-file agentic Telegram bot for TDS Project 1, Q5. Answers data-analysis
questions sent over Telegram, replying with exactly one JSON object per message.

## Architecture

```
FastAPI web app  -> GET /health        (keep-alive + sanity check)
                 -> GET /run.jsonl     (public agent log)

Background thread -> Telegram getUpdates long-poll loop
                     -> per-message: agent loop -> sendMessage(JSON)

Background thread -> self-ping /health every 10 min (free hosts idle out)
```

The agent loop gives the model one tool, `run_python`, which executes
model-written Python server-side (with pandas/numpy/requests/BeautifulSoup/
openpyxl available) and returns captured stdout. This lets the model fetch and
compute real answers instead of guessing.

## Defensive layers

- **JSON extraction**: strips code fences, finds the first balanced `{...}`
  block, and always overwrites `log_url` with the real public URL.
- **Wall-clock budget** (~210s): past this, tool calls are disabled and the
  model is forced to answer immediately — a late answer scores zero.
- **Per-chat history** (last 20 turns): keeps multi-turn context per Telegram
  chat.
- **Never crash silently**: every handler is wrapped in try/except; a reply
  that parses beats a timeout.
- **Full JSONL logging**: every question, tool call, tool result, and final
  reply is logged and served at `/run.jsonl`.
- **Model fallback chain**: tries several Gemini Flash-tier models in order,
  falling back on rate limits or errors, and logs which model actually
  answered each question.

## Setup

```bash
pip install -r requirements.txt

export TELEGRAM_BOT_TOKEN=<from @BotFather>
export LLM_API_KEY=<your Gemini API key from aistudio.google.com>
export LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

python agent.py
```

## Deploy (Render)

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn agent:app --host 0.0.0.0 --port $PORT`
- Env vars: `TELEGRAM_BOT_TOKEN`, `LLM_API_KEY`, `LLM_BASE_URL`
- Render sets `RENDER_EXTERNAL_URL` automatically; the app uses this for the
  self-ping and `log_url` unless `PUBLIC_URL` is set explicitly.

After changing env vars on Render, trigger a manual deploy — env var changes
alone do not restart the service.

## Verify

```bash
curl https://<your-service>.onrender.com/health
wget https://<your-service>.onrender.com/run.jsonl
```
