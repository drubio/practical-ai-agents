# LLM application to chat with multiple LLMs

This chapter provides a cross-framework (LangChain, LlamaIndex), dual-language (Python, JavaScript) application to interact with multiple LLMs (GPT, Claude, Gemini, Grok)

It can be run in two forms:
- In **command line** mode, with output sent to a console.
- In **web API** mode, with a web server allowing output through HTTP calls. 

## Project structure and characteristics
```
chapter_4/
├── langchain/
│   ├── llm_app.py
│   └── llm_app.js
├── llamaindex/
│   ├── llm_app.py
│   └── llm_app.js
├── utils.py
├── utils.js
├── web.py
├── web.js
````

You choose what framework to run by choosing the script language and framework subfolder:

------------------------------------------------------------------------------------
| Framework       | Python                          | JavaScript                   |
|-----------------|---------------------------------|------------------------------|
| **LangChain**   | `langchain/llm_app.py`          | `langchain/llm_app.js`       |
| **LlamaIndex**  | `llamaindex/llm_app.py`         | `llamaindex/llm_app.js`      |
------------------------------------------------------------------------------------

The following files are **shared** for both LangChain and LlamaIndex implementations, as well as other chapter exercises:

- `utils.py` / `utils.js` — shared CLI and logic
- `web.py` / `web.js` — shared web server for all frameworks

## Environment Setup
Ensure you install the dependencies of your language of choice located in the root level folder—requirements.txt or package.json. These dependencies are needed to run code in this chapter and the shared/ folder.

Once these dependencies are installed, you must also ensure the shared/ folder has access to an `.env` file with all the necessary LLM API keys. These API keys ensure your code has to LLM providers. See the shared/ folder README.md for additional details.


## Usage Modes

Run in either:

* [Command Line Mode]
### ❯ Example (Python)
```bash
python langchain/llm_app.py
python llamaindex/llm_app.py
```

### ❯ Example (JavaScript)

```bash
node langchain/llm_app.js
node llamaindex/llm_app.js
```

### ❯ Sample Session

```
What topic do you want to ask about? artificial general intelligence
Temperature (0.0-2.0, default 0.7): 0.6
Max tokens (default 1000): 500
Query ALL providers or select one? (all/one): all
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
python langchain/llm_app.py web
python llamaindex/llm_app.py web
```

### ❯ Example (JavaScript)

```bash
node langchain/llm_app.js web
node llamaindex/llm_app.js web
```

### ❯ Endpoints

| Method | Path         | Description                      |
| ------ | ------------ | -------------------------------- |
| GET    | `/`          | Service status and init messages |
| GET    | `/providers` | List initialized providers       |
| POST   | `/query`     | Query a single provider          |
| POST   | `/query-all` | Query all available providers    |

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

---

### ✅ Query All Providers

```bash
curl -X POST http://localhost:8000/query-all \
  -H "Content-Type: application/json" \
  -d '{
        "topic": "Explain the Turing Test",
        "temperature": 0.5,
        "max_tokens": 400
      }'
```

```json
{
  "success": true,
  "prompt": "Explain the Turing Test",
  "responses": {
    "openai": {
      "success": true,
      "response": "The Turing Test evaluates whether a machine can mimic human responses...",
      "model": "gpt-4o"
    },
    "anthropic": {
      "success": true,
      "response": "The Turing Test, proposed by Alan Turing...",
      "model": "claude-3-5-sonnet-20241022"
    }
  }
}
```
