# Agent application with memory and persistent chat for multiple LLMs

-This chapter extends Chapter 4's Agent application with:

1. **Memory and persistence**
2. **Memory, persistence and structured output parsing**

Both variations are available across **LangChain** and **LlamaIndex**, in **Python** and **JavaScript**, as well as CLI and web API modes.

## Project structure

```text
chapter_5/
├── langchain/
│   ├── agent_memory_persist.py
│   ├── agent_memory_persist.js
│   ├── agent_structured_output.py
│   └── agent_structured_output.js
├── llamaindex/
│   ├── agent_memory_persist.py
│   ├── agent_memory_persist.js
│   ├── agent_structured_output.py
│   └── agent_structured_output.js
└── README.md
```

## Script matrix

| Framework | Memory and Persistence(Python) | Memory and Persistence memory (JavaScript) | Structured Output (Python) | Structured Output (JavaScript) |
|---|---|---|---|---|
| **LangChain** | `langchain/agent_memory_persist.py` | `langchain/agent_memory_persist.js` | `langchain/agent_structured_output.py` | `langchain/agent_structured_output.js` |
| **LlamaIndex** | `llamaindex/agent_memory_persist.py` | `llamaindex/agent_memory_persist.js` | `llamaindex/agent_structured_output.py` | `llamaindex/agent_structured_output.js` |

## Dependencies and environment

Chapter 5 reuses Chapter 4 framework managers plus centralized shared components:
- `chapter_4/langchain/agent_app.py` / `chapter_4/langchain/agent_app.js`
- `chapter_4/llamaindex/agent_app.py` / `chapter_4/llamaindex/agent_app.js`
- `shared/llm_models.py` / `shared/llm_models.mjs`
- `shared/utils.py` / `shared/utils.mjs` and `shared/web.py` / `shared/web.mjs`
- `shared/essentials/utils.py` / `shared/essentials/utils.mjs`
- `shared/essentials/web.py` / `shared/essentials/web.mjs`

Ensure you install the dependencies for your language of choice located in the root level folder—requirements.txt or package.json—in addition to declaring LLM API keys in the shared/.env file. See the shared/ folder README.md for additional details.


## Usage

Run commands from `volume_1/chapter_5`.

### Command line mode

#### Python

```bash
python langchain/agent_memory_persist.py
python langchain/agent_structured_output.py
python llamaindex/agent_memory_persist.py
python llamaindex/agent_structured_output.py
```

#### JavaScript

```bash
node langchain/agent_memory_persist.js
node langchain/agent_structured_output.js
node llamaindex/agent_memory_persist.js
node llamaindex/agent_structured_output.js
```

### Web API mode

```bash
python langchain/agent_memory_persist.py web
node llamaindex/agent_structured_output.js web
```

## Incremental learning goal

- **Memory and Persistence** shows reusable conversation state across runs.
- **Structured Outputs** keep memory while adding JSON output parsing, so downstream code can consume stable fields (`answer`, `summary`, `keywords`, `distilled`).

## Memory-aware endpoints

In addition to base Chapter 4 endpoints (`/`, `/providers`, `/query`, `/health`), memory-capable managers expose:

| Method | Path | Description |
|---|---|---|
| GET | `/history?session_id=<id>` | Get stored turns for a session |
| POST | `/reset-memory` | Clear memory by session or clear all |

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
curl "http://localhost:8000/history?session_id=default"
```

### Reset memory

```bash
curl -X POST http://localhost:8000/reset-memory \
  -H "Content-Type: application/json" \
  -d '{"session_id": "default"}'
```

To clear all sessions use no parameters:

```bash
curl -X POST http://localhost:8000/reset-memory \
  -H "Content-Type: application/json" \
```


## Notes

- Session memory is isolated by `session_id`.
- Persistent sessions are stored under each framework's `sessions/` directory.
- Structured variants return parsed JSON in `response` and keep a short `raw_answer`/`rawAnswer` field.
