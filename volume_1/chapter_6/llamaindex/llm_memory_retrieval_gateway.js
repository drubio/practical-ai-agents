/**
 * LLM Retrieval Memory Gateway - LlamaIndex JS with selective memory replay.
 */

import { LlamaIndexLLMManager as Chapter5StructuredLlamaIndexManager } from '../../chapter_5/llamaindex/llm_memory_structured_gateway.js';
import { interactiveCli } from '../../chapter_4/utils.js';

const RETRIEVAL_TEMPLATE = `Use the selected memory snippets and answer the topic.

Selected Memory:
{memory_context}

Topic: {topic}`;

class LlamaIndexLLMManager extends Chapter5StructuredLlamaIndexManager {
    constructor(memoryEnabled = true, memoryWindow = 6) {
        super(memoryEnabled);
        this.framework = 'LlamaIndex+RetrievalMemory JS';
        this.memoryWindow = memoryWindow;
    }

    _scoreTurn(turn, queryTerms) {
        const text = String(turn?.content || '').toLowerCase();
        return queryTerms.reduce((score, term) => score + (text.includes(term) ? 1 : 0), 0);
    }

    _selectMemoryContext(provider, sessionId, topic) {
        const history = this.getHistory(provider, sessionId)?.turns || [];
        if (!history.length) return [];

        const queryTerms = topic
            .toLowerCase()
            .split(/\s+/)
            .filter((term) => term.length > 2);

        return [...history]
            .sort((a, b) => this._scoreTurn(b, queryTerms) - this._scoreTurn(a, queryTerms))
            .slice(0, this.memoryWindow);
    }

    async askQuestion(topic, provider = null, template = RETRIEVAL_TEMPLATE, maxTokens = 1000, temperature = 0.7, sessionId = 'default') {
        const resolvedProvider = this._resolveProvider(provider);
        const selectedTurns = resolvedProvider ? this._selectMemoryContext(resolvedProvider, sessionId, topic) : [];
        const memoryContext = selectedTurns.map((turn) => `- ${turn.role}: ${turn.content}`).join('\n') || '(none)';
        const prompt = template.replace('{memory_context}', memoryContext).replace('{topic}', topic);

        const result = await super.askQuestion(topic, provider, prompt, maxTokens, temperature, sessionId);
        return { ...result, selectedMemoryTurns: selectedTurns, memoryWindow: this.memoryWindow };
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
