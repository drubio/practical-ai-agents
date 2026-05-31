# Agent application to chat with multiple LLMs

This chapter provides a cross-framework (LangChain, LlamaIndex), dual-language (Python, JavaScript) agent application to interact with multiple LLMs (GPT, Claude, Gemini, Grok)

It can be run in two forms:
- In **command line** mode, with output sent to a console.
- In **web API** mode, with a web server allowing output through HTTP calls. 

## Project structure and characteristics
```
chapter_4/
├── langchain/
│   ├── agent_app.py
│   └── agent_app.js
├── llamaindex/
│   ├── agent_app.py
│   └── agent_app.js
│
shared/
├── llm_models.py / llm_models.mjs
├── utils.py / utils.mjs
├── web.py / web.mjs
├── essentials/
│   ├── utils.py / utils.mjs
│   └── web.py / web.mjs
├── .env

````

You choose what framework to run by choosing the script language and framework subfolder:

------------------------------------------------------------------------------------
| Framework       | Python                          | JavaScript                   |
|-----------------|---------------------------------|------------------------------|
| **LangChain**   | `langchain/agent_app.py`          | `langchain/agent_app.js`       |
| **LlamaIndex**  | `llamaindex/agent_app.py`         | `llamaindex/agent_app.js`      |
------------------------------------------------------------------------------------

These exercises use reusable modules in `shared/` and `shared/essentials/`; 

The shared modules are split by series and book:

- `llm_models.py` / `llm_models.mjs` — LLM model configurations used across the book series
- `utils.py` / `utils.mjs` — Utilities used across the book series, including provider/model lookup, response normalization, structured JSON parsing, logging and simple CLI helpers.
- `web.py` / `web.mjs` — Web logic used across the book series, including app/server setup, manager construction, SSE formatting, chunk streaming and capability checks.
- `essentials/utils.py` / `essentials/utils.mjs` — Utilities for this book 'Agent Essentials' , includes provider-manager base class and memory-aware interactive CLI flow.
- `essentials/web.py` / `essentials/web.mjs` — Web logic for this book 'Agent Essentials', includes status/providers/capabilities/query/query-stream/history/reset-memory endpoints.
- `.env` — Environment file with LLM API keys


## Environment Setup
Ensure you install the dependencies for your language of choice located in the root level folder—requirements.txt or package.json. These dependencies are needed to run code in this chapter, as well as shared/ folders.

Once these dependencies are installed, you must also ensure the shared/ folder has access to an `.env` file with all the necessary LLM API keys. These API keys ensure your code can access LLM providers. See the shared/ folder README.md for additional details.


## Usage Modes

Run in either:

* [Command Line Mode]
### ❯ Example (Python)
```bash
python langchain/agent_app.py
python llamaindex/agent_app.py
```

### ❯ Example (JavaScript)

```bash
node langchain/agent_app.js
node llamaindex/agent_app.js
```

### ❯ Sample Session

```
What topic do you want to ask about? artificial general intelligence
Temperature (0.0-2.0, default 0.7): 0.6
Max tokens (default 1000): 500
```
**Output:**

```
=== OpenAI GPT (LangChain) answered:
[temp: 0.6, max_tokens: 500, model: gpt-4o]
Artificial general intelligence (AGI) refers to...

=== Anthropic Claude (LangChain) answered:
[temp: 0.6, max_tokens: 500, model: claude-3-5-sonnet-20241022]
AGI is a type of AI that can perform...
```

* [Web API Mode]

**NOTE**: Only one **web server** can be active at a time, since all use **port 8000** by default.

### ❯ Example (Python)

```bash
python langchain/agent_app.py web
python llamaindex/agent_app.py web
```

### ❯ Example (JavaScript)

```bash
node langchain/agent_app.js web
node llamaindex/agent_app.js web
```

### ❯ Endpoints

| Method | Path         | Description                      |
| ------ | ------------ | -------------------------------- |
| GET    | `/`          | Service status and init messages |
| GET    | `/providers` | List initialized providers       |
| POST   | `/query`     | Query a single provider          |

---

## API Examples (`curl`)

> Output is identical between Python and JS backends.

---

### ✅ Get Status

```bash
curl http://localhost:8000/
```

```json
{
  "framework": "LangChain",
  "available_providers": ["openai", "anthropic"],
  "total_available": 2,
  "status": "healthy"
}
```

---

### ✅ Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "framework": "LangChain",
  "providers_available": 2
}
```

---

### ✅ Provider Metadata

```bash
curl http://localhost:8000/providers
```

```json
{
  "framework": "LangChain",
  "providers": [
    {
      "name": "openai",
      "display_name": "OpenAI GPT",
      "model": "gpt-4o",
      "status": "✓ Initialized successfully"
    },
    {
      "name": "anthropic",
      "display_name": "Anthropic Claude",
      "model": "claude-3-5-sonnet-20241022",
      "status": "✓ Initialized successfully"
    }
  ]
}
```

---

### ✅ Query a Single Provider

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "topic": "What is machine learning?",
        "provider": "openai",
        "temperature": 0.7,
        "max_tokens": 300
      }'
```

```json
{
  "success": true,
  "provider": "openai",
  "model": "gpt-4o",
  "response": "Machine learning is a field of AI...",
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 300
  },
  "prompt": "What is machine learning?"
}
```
