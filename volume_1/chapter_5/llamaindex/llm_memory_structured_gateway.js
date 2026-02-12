/**
 * LLM Memory Structured Gateway - LlamaIndex JS with structured JSON responses.
 */

import { LlamaIndexLLMManager as Chapter5LlamaIndexManager } from './llm_memory_persist_gateway.js';
import { interactiveCli } from '../../chapter_4/utils.js';

const STRUCTURED_TEMPLATE = `Given the topic below, provide:

1. A direct factual answer (if possible)
2. A summary of what the question is about
3. Relevant keywords
4. A distilled answer (short phrase or value-only form of the answer)

Respond in the following JSON format:
{
  "answer": "...",
  "summary": "...",
  "keywords": ["...", "..."],
  "distilled": "..."
}

Topic: {topic}`;

class LlamaIndexLLMManager extends Chapter5LlamaIndexManager {
    constructor(memoryEnabled = true) {
        super(memoryEnabled);
        this.framework = 'LlamaIndex+Structured JS';
    }

    _parseStructuredResponse(raw) {
        let content = String(raw || '').trim();
        if (content.startsWith('```json')) content = content.slice(7);
        if (content.startsWith('```')) content = content.slice(3);
        if (content.endsWith('```')) content = content.slice(0, -3);
        return JSON.parse(content.trim());
    }

    async askQuestion(topic, provider = null, template = STRUCTURED_TEMPLATE, maxTokens = 1000, temperature = 0.7, sessionId = 'default') {
        const result = await super.askQuestion(topic, provider, template, maxTokens, temperature, sessionId);

        if (!result.success) {
            return result;
        }

        const rawResponse = String(result.response ?? '');
        try {
            const parsed = this._parseStructuredResponse(rawResponse);
            return {
                ...result,
                response: parsed,
                rawAnswer: parsed.answer ?? rawResponse,
            };
        } catch (error) {
            return {
                ...result,
                success: false,
                error: `Failed to parse structured JSON response: ${error.message}`,
                response: null,
                rawResponse,
            };
        }
    }
}

async function main() {
    const args = process.argv.slice(2);
    if (args.length > 0 && args[0] === 'web') {
        const { runWebServer } = await import('../../chapter_4/web.js');
        await runWebServer(() => new LlamaIndexLLMManager(true));
    } else {
        const manager = new LlamaIndexLLMManager(true);
        await manager._checkProviders();
        await interactiveCli(manager);
    }
}

export { LlamaIndexLLMManager };

if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}
