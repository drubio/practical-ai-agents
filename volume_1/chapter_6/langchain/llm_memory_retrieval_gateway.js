/**
 * LLM Retrieval Memory Gateway - LangChain JS with selective memory replay.
 */

import { LangChainLLMManager as Chapter5StructuredLangChainManager, STRUCTURED_TEMPLATE } from '../../chapter_5/langchain/llm_memory_structured_gateway.js';
import { getDefaultModel, interactiveCli, parseStructuredJsonResponse } from '../../chapter_4/utils.js';

class LangChainLLMManager extends Chapter5StructuredLangChainManager {
    constructor(memoryEnabled = true, retrievalK = 4) {
        // Disable inherited full-history replay chain. Chapter 6 builds retrieval prompts directly.
        super(false);
        this.framework = 'LangChain+Memory+Retrieval JS';
        this.retrievalMemoryEnabled = memoryEnabled;
        this.retrievalK = Math.max(1, retrievalK);
    }

    _estimateTokens(text) {
        if (!text) return 0;
        return Math.max(1, Math.floor(String(text).length / 4));
    }

    _tokenize(text) {
        return new Set(String(text || '').toLowerCase().match(/[a-zA-Z0-9_]+/g) || []);
    }

    _scoreMessage(queryTokens, content) {
        const contentTokens = this._tokenize(content);
        if (queryTokens.size === 0 || contentTokens.size === 0) return 0;
        let score = 0;
        for (const token of queryTokens) {
            if (contentTokens.has(token)) score += 1;
        }
        return score;
    }

    _selectRetrievedMessages(topic, messages) {
        const queryTokens = this._tokenize(topic);
        const scored = [];
        messages.forEach((msg, idx) => {
            const content = String(msg?.content ?? '');
            if (!content) return;
            const score = this._scoreMessage(queryTokens, content);
            if (score > 0) scored.push({ score, idx, msg });
        });

        if (scored.length === 0) return [];

        scored.sort((a, b) => (b.score - a.score) || (a.idx - b.idx));
        const topChronological = scored
            .slice(0, this.retrievalK)
            .sort((a, b) => a.idx - b.idx);

        return topChronological.map(({ score, msg }) => ({
            role: msg?._getType?.() ?? msg?.getType?.() ?? msg?.type ?? 'unknown',
            content: String(msg?.content ?? ''),
            relevance_score: score,
        }));
    }

    async askQuestion(topic, provider = null, template = STRUCTURED_TEMPLATE, maxTokens = 1000, temperature = 0.7, sessionId = 'default') {
        const resolvedProvider = this._resolveProvider(provider);
        const basePrompt = template.replace('{topic}', topic);

        if (!resolvedProvider) {
            return {
                success: false,
                error: 'No providers available',
                provider: 'none',
                model: 'none',
                prompt: basePrompt,
                response: null,
            };
        }

        let messages = [];
        if (this.retrievalMemoryEnabled) {
            const history = this._getHistory(resolvedProvider, sessionId);
            messages = await history.getMessages();
        }

        const retrieved = this.retrievalMemoryEnabled ? this._selectRetrievedMessages(topic, messages) : [];
        const retrievedContext = retrieved.map((item) => `[${item.role}] ${item.content}`).join('\n');
        const retrievalAugmentedTopic = retrievedContext
            ? `Relevant memory snippets:\n${retrievedContext}\n\nCurrent user topic: ${topic}`
            : topic;
        const retrievalPrompt = template.replace('{topic}', retrievalAugmentedTopic);

        const fullHistoryContext = messages
            .map((msg) => `[${msg?._getType?.() ?? msg?.getType?.() ?? msg?.type ?? 'unknown'}] ${String(msg?.content ?? '')}`)
            .join('\n');
        const promptWithoutRetrieval = fullHistoryContext
            ? `${fullHistoryContext}\n\nCurrent user topic: ${topic}`
            : topic;

        const tokensWithRetrieval = this._estimateTokens(retrievalPrompt);
        const tokensWithoutRetrieval = this._estimateTokens(promptWithoutRetrieval);
        const estimatedSaved = Math.max(0, tokensWithoutRetrieval - tokensWithRetrieval);
        const reductionPercent = tokensWithoutRetrieval > 0
            ? Number(((estimatedSaved / tokensWithoutRetrieval) * 100).toFixed(2))
            : 0;

        try {
            const client = this._createClient(resolvedProvider, temperature, maxTokens);
            const result = await client.invoke(this._buildMessages(retrievalPrompt));
            const rawResponse = this._extractText(resolvedProvider, result);

            const responseMetadata = result?.response_metadata ?? result?.responseMetadata ?? null;
            const usageMetadata = result?.usage_metadata ?? result?.usageMetadata ?? null;
            const tokenUsage = this._extractTokenUsage(responseMetadata, usageMetadata);

            const metadataPayload = {
                provider: resolvedProvider,
                model: getDefaultModel(resolvedProvider),
                sessionId,
                temperature,
                maxTokens,
                response_metadata: responseMetadata,
                usage_metadata: usageMetadata,
            };

            const parsed = parseStructuredJsonResponse(rawResponse);
            parsed.metadata = {
                ...(parsed.metadata || {}),
                ...this._buildMetadata(metadataPayload, rawResponse),
                retrieval: {
                    history_messages_available: messages.length,
                    retrieved_messages_count: retrieved.length,
                    retrieved_messages: retrieved,
                    tokens_with_memory_retrieval: tokensWithRetrieval,
                    tokens_without_memory_retrieval: tokensWithoutRetrieval,
                    estimated_tokens_saved: estimatedSaved,
                    estimated_token_reduction_percent: reductionPercent,
                },
            };

            if (this.retrievalMemoryEnabled) {
                const history = this._getHistory(resolvedProvider, sessionId);
                await history.addUserMessage(topic);
                await history.addAIMessage(rawResponse);
            }

            return {
                success: true,
                provider: resolvedProvider,
                model: getDefaultModel(resolvedProvider),
                prompt: retrievalPrompt,
                response: parsed,
                rawAnswer: parsed.answer ?? rawResponse,
                temperature,
                maxTokens,
                sessionId,
                ...(responseMetadata != null ? { response_metadata: responseMetadata } : {}),
                ...(usageMetadata != null ? { usage_metadata: usageMetadata } : {}),
                ...(tokenUsage != null ? { token_usage: tokenUsage } : {}),
            };
        } catch (error) {
            return {
                success: false,
                provider: resolvedProvider,
                model: getDefaultModel(resolvedProvider),
                prompt: retrievalPrompt,
                error: error.message,
                response: null,
                temperature,
                maxTokens,
                sessionId,
            };
        }
    }
}

async function main() {
    const args = process.argv.slice(2);
    if (args.length > 0 && args[0] === 'web') {
        const { runWebServer } = await import('../../chapter_4/web.js');
        await runWebServer(() => new LangChainLLMManager(true));
    } else {
        const manager = new LangChainLLMManager(true);
        await manager._checkProviders();
        await interactiveCli(manager);
    }
}

export { LangChainLLMManager };

if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}
