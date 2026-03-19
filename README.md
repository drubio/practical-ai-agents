# Software that works with AI

*A practical guide to modern AI integration*

This three-part series offers a structured, hands-on approach to building software applications that work with large language models (LLMs). Whether you're a developer, engineer or technical lead, this guide helps you design and implement AI-powered software systems with clarity and practicality.

---

## 📘 Volumes

### [Essential LLM Applications – Fundamentals, APIs and UIs](https://www.amazon.com/dp/B0GRG4Y71Q?tag=github-drubio-20) _(Volume 1)_

Covers how LLMs work, their evolution and how to access them via APIs (OpenAI, Anthropic, Google, xAI). Includes building a complete app using:

- Step-by-step examples for developers new to LLMs
- LangChain and LlamaIndex for basic chats with LLMs
- Assistant UI components and vanilla React


### Scale LLM Applications – Agents, Routing and Infrastructure _(Volume 2)_ _(coming soon)_

Explores:


- Agent ReAct ("Reasoning + Acting") pattern, routing (tool and models) and memory management
- Multi-agent orchestration using frameworks and managing model infrastructure <!---with LiteLLM and OpenRouter--><!---like CrewAI-->
- Designing scalable, fault-tolerant systems with context and robustness in mind

### Advanced LLM Applications – RAG, Personalization and MCP _(Volume 3)_ _(coming soon)_

Focuses on:

- Retrieval-Augmented Generation (RAG) pipelines
- Real-time user interaction with LLMs
- Applying the Model Context Protocol (MCP) for dynamic, extensible applications

---

## About the Code

This repository includes companion code and examples for all three volumes. Code is organized to match the topics covered in each volume. Each folder corresponds to a chapter described in the book.

You're welcome to explore, modify and adapt the examples for your own projects.

### Root shared dependency files

This repository root includes two dependency manifests:

- `requirements.txt` contains the common Python dependencies used by shared helpers and by chapters that rely on this code.
- `package.json` contains the common TypeScript/JavaScript dependencies used by shared helpers and by chapters that rely on this code.

NOTE: *Volume 1*: *Chapters 1 through 3* and *Chapter 8* have their own dependency files in their chapter folders. Each of these chapters should run their own self-contained install.

---

## Support the Work

If you find this useful, please consider [purchasing Volume 1](https://www.amazon.com/dp/B0GRG4Y71Q?tag=github-drubio-20) and sharing it with others. Your support makes continued writing and open-source work possible.
