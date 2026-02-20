# Chapter 6: Retrieval Memory Gateway

Chapter 6 upgrades the Chapter 5 memory gateways from **full-history replay** to **retrieval-based memory**.

It includes:
- **LangChain** and **LlamaIndex** implementations
- **Python** and **JavaScript** versions for each framework
- CLI + Web modes (same as earlier chapters)

## What's new vs earlier chapters

### vs Chapter 4 (basic gateway)
- Adds conversational memory support through Chapter 5 managers.
- Adds retrieval selection so prompts include only relevant prior turns.

### vs Chapter 5 (memory + persistence + structured output)
- Stops replaying the entire conversation history into each prompt.
- Uses BM25 retrieval to select top-`k` relevant snippets for prompt injection.
- Combines BM25 with lexical-overlap gating, then falls back to overlap-only matching when BM25 is weak.
- Uses framework/native tokenizers for token-count estimates in retrieval metadata (`retrieved_messages_count`, token savings estimates, etc.); JS uses a provider-agnostic BPE baseline tokenizer.
- Keeps persistent session memory behavior inherited from Chapter 5.

### Robust structured-response behavior (Py + JS)
- If the CLI passes `'{topic}'`, Chapter 6 retrieval scripts normalize to `STRUCTURED_TEMPLATE` so structured JSON instructions are still used.
- If a provider returns non-JSON text, parsers now fall back to a safe structured payload instead of hard-failing.

## Project structure

```text
chapter_6/
├── langchain/
│   ├── llm_memory_retrieval_gateway.py
│   └── llm_memory_retrieval_gateway.js
├── llamaindex/
│   ├── llm_memory_retrieval_gateway.py
│   └── llm_memory_retrieval_gateway.js
└── README.md
```

## Script matrix

| Framework | Python | JavaScript |
|---|---|---|
| LangChain | `langchain/llm_memory_retrieval_gateway.py` | `langchain/llm_memory_retrieval_gateway.js` |
| LlamaIndex | `llamaindex/llm_memory_retrieval_gateway.py` | `llamaindex/llm_memory_retrieval_gateway.js` |

## Dependencies and environment

Chapter 6 reuses:
- Chapter 4 provider setup, shared CLI/web utilities, and `.env`
- Chapter 5 memory/session persistence foundations

Set keys in `volume_1/chapter_4/.env`:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
XAI_API_KEY=your-xai-key
```

## Usage

Run from `volume_1/chapter_6`.

### CLI mode

#### Python

```bash
python langchain/llm_memory_retrieval_gateway.py
python llamaindex/llm_memory_retrieval_gateway.py
```

#### JavaScript

```bash
node langchain/llm_memory_retrieval_gateway.js
node llamaindex/llm_memory_retrieval_gateway.js
```

### Web mode

```bash
python langchain/llm_memory_retrieval_gateway.py web
python llamaindex/llm_memory_retrieval_gateway.py web
node langchain/llm_memory_retrieval_gateway.js web
node llamaindex/llm_memory_retrieval_gateway.js web
```

## Retrieval metadata

Successful responses include structured output plus retrieval diagnostics in metadata, including:
- `history_messages_available`
- `retrieved_messages_count`
- `retrieved_messages`
- `tokens_with_memory_retrieval`
- `tokens_without_memory_retrieval`
- `estimated_tokens_saved`
- `estimated_token_reduction_percent`

## Notes

- Session memory remains isolated by provider + session id.
- Retrieval memory still persists turns to session files through Chapter 5 persistence hooks.
- Retrieval reduces prompt size pressure while preserving relevant context.
