# LLM application with memory and persistent chat for multiple LLMs

-This chapter extends Chapter 4's LLM application with:

1. **Memory and persistence**
2. **Memory, persistence and structured output parsing**

Both variations are available across **LangChain** and **LlamaIndex**, in **Python** and **JavaScript**, as well as CLI and web API modes.

## Project structure

```text
chapter_5/
├── langchain/
│   ├── llm_memory_persist.py
│   ├── llm_memory_persist.js
│   ├── llm_structured_output.py
│   └── llm_structured_output.js
├── llamaindex/
│   ├── llm_memory_persist.py
│   ├── llm_memory_persist.js
│   ├── llm_structured_output.py
│   └── llm_structured_output.js
└── README.md
```

## Script matrix

| Framework | Memory and Persistence(Python) | Memory and Persistence memory (JavaScript) | Structured Output (Python) | Structured Output (JavaScript) |
|---|---|---|---|---|
| **LangChain** | `langchain/llm_memory_persist.py` | `langchain/llm_memory_persist.js` | `langchain/llm_structured_output.py` | `langchain/llm_structured_output.js` |
| **LlamaIndex** | `llamaindex/llm_memory_persist.py` | `llamaindex/llm_memory_persist.js` | `llamaindex/llm_structured_output.py` | `llamaindex/llm_structured_output.js` |

## Dependencies and environment

Chapter 5 reuses Chapter 4 and shared components:
- `chapter_4/utils.py` / `chapter_4/utils.js`
- `chapter_4/web.py` / `chapter_4/web.js`
- `shared/llm_models.py` / `shared/llm_models.mjs`
- `shared/utils.py` / `shared/utils.mjs`
- `shared/web.py` / `shared/web.mjs`

Ensure you install the dependencies for your language of choice located in the root level folder—requirements.txt or package.json—in addition to declaring LLM API keys in the shared/.env file. See the shared/ folder README.md for additional details.


## Usage

Run commands from `volume_1/chapter_5`.

### Command line mode

#### Python

```bash
python langchain/llm_memory_persist.py
python langchain/llm_structured_output.py
python llamaindex/llm_memory_persist.py
python llamaindex/llm_structured_output.py
```

#### JavaScript

```bash
node langchain/llm_memory_persist.js
node langchain/llm_structured_output.js
node llamaindex/llm_memory_persist.js
node llamaindex/llm_structured_output.js
```

### Web API mode

```bash
python langchain/llm_memory_persist.py web
node llamaindex/llm_structured_output.js web
```

## Incremental learning goal

- **Memory and Persistence** shows reusable conversation state across runs.
- **Structured Outputs** keep memory while adding JSON output parsing, so downstream code can consume stable fields (`answer`, `summary`, `keywords`, `distilled`).

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
