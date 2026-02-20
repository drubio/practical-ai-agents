# LLM Tools Gateway

This chapter extends Chapter 6's structured gateways with **tool calling**.

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
│   ├── llm_tools_gateway.py
│   └── llm_tools_gateway.js
├── llamaindex/
│   ├── llm_tools_gateway.py
│   └── llm_tools_gateway.js
├── tools.py
├── tools.js
├── requirements.txt
├── package.json
└── README.md
```

## Script matrix

| Framework | Python | JavaScript |
|---|---|---|
| **LangChain** | `langchain/llm_tools_gateway.py` | `langchain/llm_tools_gateway.js` |
| **LlamaIndex** | `llamaindex/llm_tools_gateway.py` | `llamaindex/llm_tools_gateway.js` |

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
python langchain/llm_tools_gateway.py
python llamaindex/llm_tools_gateway.py
```

#### JavaScript

```bash
node langchain/llm_tools_gateway.js
node llamaindex/llm_tools_gateway.js
```

### Web API mode

```bash
python langchain/llm_tools_gateway.py web
python llamaindex/llm_tools_gateway.py web
node langchain/llm_tools_gateway.js web
node llamaindex/llm_tools_gateway.js web
```


## Integration with Chapters 4-6

Each Chapter 7 gateway now subclasses its Chapter 6 retrieval manager directly, so every Chapter 6 behavior remains active:

- provider setup + CLI/web helpers from Chapter 4
- session memory persistence from Chapter 5
- retrieval-based memory selection and retrieval diagnostics from Chapter 6
- Chapter 7 two-step tool loop layered on top

This keeps the chapter progression consistent across:
- LangChain Python
- LangChain JavaScript
- LlamaIndex Python
- LlamaIndex JavaScript

## Tool orchestration pattern

All Chapter 7 gateways follow the same two-step JSON tool loop:

1. Model returns JSON with:
   - `tool_call`: either `null` or `{ "name": "...", "arguments": { ... } }`
   - `final_answer`: short draft answer
2. Gateway executes tool locally when `tool_call` is present.
3. Gateway asks model for a final JSON response using tool output.

### Expected response shape

```json
{
  "tool_call": null,
  "tool_output": "...",
  "final_answer": "..."
}
```

- If no tool is needed, `tool_call` is `null` and `tool_output` remains `null`.
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
- If a model returns non-JSON output, gateways surface a parsing error with the raw response context.
