/**
 * LLM Retrieval Memory Gateway - LlamaIndex JS with selective memory replay.
 */

import { LlamaIndexLLMManager as Chapter5StructuredLlamaIndexManager, STRUCTURED_TEMPLATE } from '../../chapter_5/llamaindex/llm_memory_structured_gateway.js';
import { getDefaultModel, interactiveCli, parseStructuredJsonResponse } from '../../chapter_4/utils.js';

class LlamaIndexLLMManager extends Chapter5StructuredLlamaIndexManager {
    constructor(memoryEnabled = true, retrievalK = 4) {
        // Disable inherited full-memory chat engine replay. Chapter 6 builds retrieval prompts directly.
        super(false);
        this.framework = 'LlamaIndex+Memory+Retrieval JS';
        this.retrievalMemoryEnabled = memoryEnabled;
        this.retrievalK = Math.max(1, retrievalK);
    }

    _estimateTokens(text) {
        if (!text) return 0;
        return Math.max(1, Math.floor(String(text).length / 4));
    }

    _tokenize(text) {
        const stopWords = new Set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from', 'how',
            'i', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was',
            'we', 'what', 'when', 'where', 'which', 'who', 'why', 'with', 'you',
        ]);
        const tokens = String(text || '').toLowerCase().match(/[a-zA-Z0-9_]+/g) || [];
        return new Set(tokens.filter((token) => token.length > 2 && !stopWords.has(token)));
    }

    _isFollowUpQuery(topic) {
        const normalized = String(topic || '').trim().toLowerCase();
        const followUpPrefixes = [
            'what about',
            'how about',
            'what else',
            'and what',
            'and how',
            'also',
            'follow up',
        ];
        return followUpPrefixes.some((prefix) => normalized.startsWith(prefix));
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

    async _getMemoryMessages(provider, sessionId) {
        const memory = this._getMemory(provider, sessionId);
        const messagePayload = await memory.get({ type: 'llamaindex' });
        return Array.isArray(messagePayload) ? messagePayload : [];
    }

    _selectRetrievedMessages(topic, messages) {
        if (this._isFollowUpQuery(topic)) {
            return messages
                .slice(-this.retrievalK)
                .filter((msg) => String(msg?.content ?? '').length > 0)
                .map((msg) => ({
                    role: msg?._getType?.() ?? msg?.getType?.() ?? msg?.type ?? msg?.role ?? 'unknown',
                    content: String(msg?.content ?? ''),
                    relevance_score: 1,
                }));
        }

        const queryTokens = this._tokenize(topic);
        if (queryTokens.size === 0) return [];

        const scored = [];

        messages.forEach((msg, idx) => {
            const content = String(msg?.content ?? '');
            if (!content) return;
            const score = this._scoreMessage(queryTokens, content);
            const overlapRatio = score / queryTokens.size;
            if (score >= 2 || overlapRatio >= 0.5) scored.push({ score, idx, msg });
        });

        if (scored.length === 0) return [];

        scored.sort((a, b) => (b.score - a.score) || (a.idx - b.idx));
        const topChronological = scored
            .slice(0, this.retrievalK)
            .sort((a, b) => a.idx - b.idx);

        return topChronological.map(({ score, msg }) => ({
            role: String(msg?.role ?? 'unknown'),
            content: String(msg?.content ?? ''),
            relevance_score: score,
        }));
    }

    async askQuestion(topic, provider = null, template = STRUCTURED_TEMPLATE, maxTokens = 1000, temperature = 0.7, sessionId = 'default') {
        const effectiveTemplate = template === '{topic}' ? STRUCTURED_TEMPLATE : template;
        const resolvedProvider = this._resolveProvider(provider);
        const basePrompt = effectiveTemplate.replace('{topic}', topic);

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

        const messages = this.retrievalMemoryEnabled
            ? await this._getMemoryMessages(resolvedProvider, sessionId)
            : [];
        const retrieved = this.retrievalMemoryEnabled
            ? this._selectRetrievedMessages(topic, messages)
            : [];

        const retrievedContext = retrieved.map((item) => `[${item.role}] ${item.content}`).join('\n');
        const retrievalAugmentedTopic = retrievedContext
            ? `Relevant memory snippets:\n${retrievedContext}\n\nCurrent user topic: ${topic}`
            : topic;
        const retrievalPrompt = effectiveTemplate.replace('{topic}', retrievalAugmentedTopic);

        const fullHistoryContext = messages
            .map((msg) => `[${String(msg?.role ?? 'unknown')}] ${String(msg?.content ?? '')}`)
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
            const result = await client.chat({
                messages: [{ role: 'user', content: retrievalPrompt }],
            });
            const rawResponse = this._extractText(result);

            const metadataPayload = {
                provider: resolvedProvider,
                model: getDefaultModel(resolvedProvider),
                sessionId,
                temperature,
                maxTokens,
            };

            let parsed;
            try {
                parsed = parseStructuredJsonResponse(rawResponse);
            } catch {
                parsed = {
                    answer: rawResponse,
                    summary: 'Model returned plain-text output instead of strict JSON.',
                    keywords: [],
                    distilled: rawResponse,
                    metadata: {
                        confidence: 'low',
                        notes: 'Structured parser fallback applied for non-JSON response.',
                    },
                };
            }

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
                await this._appendToMemory(resolvedProvider, sessionId, 'user', topic);
                await this._appendToMemory(resolvedProvider, sessionId, 'assistant', rawResponse);
                await this._persistMemory(resolvedProvider, sessionId);
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
