# Shared utilities

The `shared/` folder contains repository-level helper modules that are reused by the multi-chapter application examples in this repository.

## Purpose

These files centralize common logic so later chapters can avoid duplicating the same implementation details across Python and JavaScript examples.

## What's here

- `utils.py` / `utils.mjs`: shared helper logic used across chapters.
- `web.py` / `web.mjs`: shared web server helpers for examples that expose HTTP APIs.
- `llm_models.py` / `llm_models.mjs`: shared model/provider configuration and initialization helpers.
- `__init__.py`: package marker for Python imports.


## Environment Setup

For each language, install the dependencies in the root level folder (one level up from here)

### ❯ Example (Python)
   ```bash
   cd ../
   pip install -r requirements # or a pip alternative like poetry or uv
````

### ❯ Example (JavaScript/TypeScript)
   ```bash
   cd ../
   npm install
````

### Create an `.env` file and place it here with the different provider LLM API keys:

   ```env
   ANTHROPIC_API_KEY="sk-ant-api03-xxxxxx"
   OPENAI_API_KEY="sk-proj-xxxx-xxxx-xxxx"
   GOOGLE_API_KEY="AIzaSyBAKuqxxxxxxxxxxxx"
   XAI_API_KEY="xai-UXxxxxxxx"
   ```

### You're set! The chapters that require this shared code to run are ready!

## Which chapters use this shared folder?

The shared code and configurations here are intended for the exercises in:

- **Volume 1, Chapters 4 through 7**
- **Volume 2, All chapters**
- **Volume 3, All chapters**
