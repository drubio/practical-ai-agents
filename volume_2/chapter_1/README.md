# Volume 2 · Chapter 1 — Shared utilties for agents

This chapter is a **shared utility layer** for all chapters in Volume 2. There are **NO** exercises to run directly.

## Shared resource usage

Chapter 1 uses the repository-level `shared/` resources and the centralized dependency setup from the repository root.

Use:

- Root `requirements.txt` for shared Python dependencies
- Root `package.json` for shared JavaScript dependencies
- `shared/` modules for common LLM/provider, utility, and web helpers

## Chapter structure

```text
chapter_1/
├── tools.py
├── tools.js
├── utils.py
├── utils.js
└── package.json (runtime dependencies for JS examples)
```

## Setup

Install shared dependencies from the repository root, then run exercises in each of the different chapters in `volume_2/`.
