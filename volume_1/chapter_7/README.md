# LLM application with tools

This chapter extends Chapter 6's features with **tool calling**.

It provides cross-framework (**LangChain**, **LlamaIndex**) and dual-language (**Python**, **JavaScript**) implementations that:

- Reuse Chapter 4 base provider/client setup and shared CLI/Web helpers
- Reuse Chapter 5 memory + persistence base behavior through Chapter 6 managers
- Reuse Chapter 6 retrieval-memory + structured-response pattern
- Add Chapter 7 tool orchestration (model decides tool call, tool executes, model synthesizes final answer)

Just like earlier chapters, each script can run in:
- **Command line mode**
- **Web API mode**

## Project structure

```text
chapter_7/
├── langchain/
│   ├── llm_tools.py
│   └── llm_tools.js
├── llamaindex/
│   ├── llm_tools.py
│   └── llm_tools.js
├── tools.py
├── tools.js
├── requirements.txt
├── package.json
└── README.md
```

## Script matrix

| Framework | Python | JavaScript |
|---|---|---|
| **LangChain** | `langchain/llm_tools.py` | `langchain/llm_tools.js` |
| **LlamaIndex** | `llamaindex/llm_tools.py` | `llamaindex/llm_tools.js` |

## Dependencies and environment

Chapter 7 builds on Chapter 4/5/6:

- Chapter 4 shared utilities and web server helpers
- Chapter 5 memory + persistence foundations
- Chapter 6 retrieval-memory manager classes
- Chapter 7 tool utilities (`tools.py`, `tools.js`)

Set API keys in `volume_1/chapter_4/.env`:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
XAI_API_KEY=your-xai-key
```

Install dependencies:

### Python

```bash
pip install -r requirements.txt
```

### JavaScript

```bash
npm install
```

## Usage

Run commands from `volume_1/chapter_7`.

### Command line mode

#### Python

```bash
python langchain/llm_tools.py
python llamaindex/llm_tools.py
```

#### JavaScript

```bash
node langchain/llm_tools.js
node llamaindex/llm_tools.js
```

### Web API mode

```bash
python langchain/llm_tools.py web
python llamaindex/llm_tools.py web
node langchain/llm_tools.js web
node llamaindex/llm_tools.js web
```

## Tool orchestration pattern

All Chapter 7 applications follow the same two-step JSON tool loop:

1. Model returns JSON with:
   - `tool_calls`: an array of tool calls, each shaped like `{ "name": "...", "arguments": { ... }, "output": null }`
   - `final_answer`: short draft answer
2. Application executes tools locally when `tool_calls` entries are present.
3. Application asks model for a final JSON response that keeps the same `tool_calls` array and fills in each `output`.

### Expected response shape

```json
{
  "tool_calls": [{
    "name": "tool_name",
    "arguments": {"arg": "value"},
    "output": "serialized tool output"
  }],
  "final_answer": "..."
}
```

- If no tool is needed, `tool_calls` is an empty array.
- On success, `raw_answer`/`rawAnswer` mirrors the final answer text.

## Included tool utilities

Current tools shared by Python/JS utilities:

- `get_wikipedia_evidence_pack` — fetch Wikipedia summary + references + Wikimedia media

Both utility modules expose:

- tool definitions metadata (`TOOL_DEFINITIONS`)
- a dispatcher (`run_tool` in Python / `runTool` in JS)
- a prompt helper (`build_tools_prompt` in Python / `buildToolsPrompt` in JS)

## Notes

- Tool contracts are intentionally simple and framework-agnostic for easy extension in later chapters.
- Session memory compatibility and provider handling continue to come from inherited chapter managers.
- If a model returns non-JSON output, application surface a parsing error with the raw response context.
