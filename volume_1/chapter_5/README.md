# LLM Memory Gateway

-This chapter extends Chapter 4's universal LLM gateway with:

1. **Persistent memory gateway** (default memory baseline)
2. **Persistent memory + structured output parsing**

Both variations are available across **LangChain** and **LlamaIndex**, in **Python** and **JavaScript**, as well as CLI and Web API modes.

## Project structure

```text
chapter_5/
├── langchain/
│   ├── llm_memory_persist_gateway.py
│   ├── llm_memory_persist_gateway.js
│   ├── llm_memory_structured_gateway.py
│   └── llm_memory_structured_gateway.js
├── llamaindex/
│   ├── llm_memory_persist_gateway.py
│   ├── llm_memory_persist_gateway.js
│   ├── llm_memory_structured_gateway.py
│   └── llm_memory_structured_gateway.js
└── README.md
```

## Script matrix

| Framework | Persistent memory (Python) | Persistent memory (JavaScript) | Structured + memory (Python) | Structured + memory (JavaScript) |
|---|---|---|---|---|
| **LangChain** | `langchain/llm_memory_persist_gateway.py` | `langchain/llm_memory_persist_gateway.js` | `langchain/llm_memory_structured_gateway.py` | `langchain/llm_memory_structured_gateway.js` |
| **LlamaIndex** | `llamaindex/llm_memory_persist_gateway.py` | `llamaindex/llm_memory_persist_gateway.js` | `llamaindex/llm_memory_structured_gateway.py` | `llamaindex/llm_memory_structured_gateway.js` |

## Dependencies and environment

Chapter 5 reuses Chapter 4 shared components:
- `chapter_4/utils.py` / `chapter_4/utils.js`
- `chapter_4/web.py` / `chapter_4/web.js`
- `chapter_4/requirements.txt` / `chapter_4/package.json`
- `chapter_4/.env`

Ensure API keys are set in `chapter_4/.env`:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
XAI_API_KEY=your-xai-key
```

## Usage

Run commands from `volume_1/chapter_5`.

### Command line mode

#### Python

```bash
python langchain/llm_memory_persist_gateway.py
python langchain/llm_memory_structured_gateway.py
python llamaindex/llm_memory_persist_gateway.py
python llamaindex/llm_memory_structured_gateway.py
```

#### JavaScript

```bash
node langchain/llm_memory_persist_gateway.js
node langchain/llm_memory_structured_gateway.js
node llamaindex/llm_memory_persist_gateway.js
node llamaindex/llm_memory_structured_gateway.js
```

### Web API mode

```bash
python langchain/llm_memory_persist_gateway.py web
node llamaindex/llm_memory_structured_gateway.js web
```

## Incremental learning goal

- **Persistent memory gateways** show reusable conversation state across runs.
- **Structured gateways** keep memory while adding JSON output parsing, so downstream code can consume stable fields (`answer`, `summary`, `keywords`, `distilled`).

## Memory-aware endpoints

In addition to base Chapter 4 endpoints (`/`, `/providers`, `/query`, `/query-all`, `/health`), memory-capable managers expose:

| Method | Path | Description |
|---|---|---|
| GET | `/history?provider=<name>&session_id=<id>` | Get stored turns for a provider/session |
| POST | `/reset-memory` | Clear memory by provider/session or clear all |

### Query with session context

Use `session_id` in `/query` requests so consecutive calls share context:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "topic": "Twinkle, Twinkle, Little",
        "provider": "openai",
        "session_id": "default"
      }'
```

### Read session history

```bash
curl "http://localhost:8000/history?provider=openai&session_id=default"
```

### Reset memory

```bash
curl -X POST http://localhost:8000/reset-memory \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai", "session_id": "default"}'
```

To clear all sessions use no parameters:

```bash
curl -X POST http://localhost:8000/reset-memory \
  -H "Content-Type: application/json" \
```


## Notes

- Session memory is isolated by `provider` + `session_id`.
- Persistent sessions are stored under each framework's `sessions/` directory.
- Structured variants return parsed JSON in `response` and keep a short `raw_answer`/`rawAnswer` field.
