/**
 * Agent Selective Memory Retrieval - LangChain JS with BM25.
 */

import { LangChainLLMManager as Chapter5StructuredLangChainManager, STRUCTURED_TEMPLATE } from '../../chapter_5/langchain/agent_structured_output.js';
import { interactiveCli, parseStructuredJsonResponse, printCliHelp } from '../../../shared/essentials/utils.mjs';
import { Document } from '@langchain/core/documents';
import { BM25Retriever } from '@langchain/community/retrievers/bm25';
import { getEncoding } from 'js-tiktoken';

class LangChainLLMManager extends Chapter5StructuredLangChainManager {
    constructor(memoryEnabled = true, retrievalK = 4) {
        // Disable inherited full-history replay chain. Chapter 6 builds retrieval prompts directly.
        super(false);
        this.framework = 'LangChain Memory+Retrieval JS';
        this.retrievalMemoryEnabled = memoryEnabled;
        this.retrievalK = Math.max(1, retrievalK);
        // Provider-agnostic tokenizer baseline (GPT-2 BPE), without model/provider mapping.
        this.tokenizer = getEncoding('gpt2');
    }

    _estimateTokens(text) {
        if (!text) return 0;
        return this.tokenizer.encode(String(text)).length;
    }

    _tokenize(text) {
        const stopWords = new Set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from', 'how',
            'i', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was',
            'we', 'what', 'when', 'where', 'which', 'who', 'why', 'with', 'you',
        ]);
        const tokens = String(text || '').toLowerCase().match(/[a-z0-9_]+/g) || [];
        return new Set(tokens.filter((token) => token.length > 2 && !stopWords.has(token)));
    }

    _overlapScore(queryTokens, content) {
        const contentTokens = this._tokenize(content);
        if (queryTokens.size === 0 || contentTokens.size === 0) return 0;
        let score = 0;
        for (const token of queryTokens) {
            if (contentTokens.has(token)) score += 1;
        }
        return score;
    }

    async _selectRetrievedMessages(topic, messages) {
        const queryTokens = this._tokenize(topic);
        if (queryTokens.size === 0) return [];

        const docs = messages
            .map((msg, idx) => {
                const content = String(msg?.content ?? '');
                if (!content) return null;
                return new Document({
                    pageContent: content,
                    metadata: {
                        idx,
                        role: msg?._getType?.() ?? msg?.getType?.() ?? msg?.type ?? msg?.role ?? 'unknown',
                    },
                });
            })
            .filter(Boolean);

        if (docs.length > 0) {
            const retriever = BM25Retriever.fromDocuments(docs, { k: this.retrievalK, includeScore: true });
            const retrievedDocs = await retriever.invoke(topic);
            const strongDocs = retrievedDocs.filter((doc) => {
                const bm25 = Number(doc?.metadata?.bm25Score ?? 0);
                if (bm25 <= 0) return false;
                const overlap = this._overlapScore(queryTokens, String(doc?.pageContent ?? ''));
                const overlapRatio = overlap / queryTokens.size;
                return overlap >= 2 || overlapRatio >= 0.4;
            });

            if (strongDocs.length > 0) {
                return strongDocs
                    .sort((a, b) => (a?.metadata?.idx ?? 0) - (b?.metadata?.idx ?? 0))
                    .map((doc) => ({
                        role: doc?.metadata?.role ?? 'unknown',
                        content: String(doc?.pageContent ?? ''),
                        relevance_score: Number(doc?.metadata?.bm25Score ?? 0),
                    }));
            }
        }

        const fallbackScored = [];
        messages.forEach((msg, idx) => {
            const content = String(msg?.content ?? '');
            if (!content) return;
            const score = this._overlapScore(queryTokens, content);
            const overlapRatio = score / queryTokens.size;
            if (score >= 2 || overlapRatio >= 0.4) fallbackScored.push({ score, idx, msg });
        });

        return fallbackScored
            .sort((a, b) => (b.score - a.score) || (a.idx - b.idx))
            .slice(0, this.retrievalK)
            .sort((a, b) => a.idx - b.idx)
            .map(({ score, msg }) => ({
                role: msg?._getType?.() ?? msg?.getType?.() ?? msg?.type ?? msg?.role ?? 'unknown',
                content: String(msg?.content ?? ''),
                relevance_score: score,
            }));
    }

    async askQuestion(topic, provider = null, template = STRUCTURED_TEMPLATE, maxTokens = 1000, temperature = 0.7, sessionId = 'default') {
        const effectiveTemplate = template === '{topic}' ? STRUCTURED_TEMPLATE : template;
        const resolvedProvider = this.resolveModelIdentifier(provider);
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

        const modelConfig = this.resolveModelConfig(resolvedProvider);

        let messages = [];
        if (this.retrievalMemoryEnabled) {
            const history = this._getHistory(sessionId);
            messages = await history.getMessages();
        }

        const retrieved = this.retrievalMemoryEnabled ? await this._selectRetrievedMessages(topic, messages) : [];
        const retrievedContext = retrieved.map((item) => `[${item.role}] ${item.content}`).join('\n');
        const retrievalAugmentedTopic = retrievedContext
            ? `Relevant memory snippets:\n${retrievedContext}\n\nCurrent user topic: ${topic}`
            : topic;
        const retrievalPrompt = effectiveTemplate.replace('{topic}', retrievalAugmentedTopic);

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
            const model = this._createModel(resolvedProvider, temperature, maxTokens);
            const result = await model.invoke(this._buildMessages(retrievalPrompt));
            const rawResponse = this._extractText(resolvedProvider, result);

            const responseMetadata = result?.response_metadata ?? result?.responseMetadata ?? null;
            const usageMetadata = result?.usage_metadata ?? result?.usageMetadata ?? null;
            const tokenUsage = this._extractTokenUsage(responseMetadata, usageMetadata);

            const metadataPayload = {
                provider: modelConfig.provider,
                model: modelConfig.model,
                modelIdentifier: modelConfig.name,
                sessionId,
                temperature,
                maxTokens,
                response_metadata: responseMetadata,
                usage_metadata: usageMetadata,
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
                const history = this._getHistory(sessionId);
                await history.addUserMessage(topic);
                await history.addAIMessage(rawResponse);
            }

            return {
                success: true,
                provider: modelConfig.provider,
                model: modelConfig.model,
                modelIdentifier: modelConfig.name,
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
                provider: modelConfig.provider,
                model: modelConfig.model,
                modelIdentifier: modelConfig.name,
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
    if (args.includes('-h') || args.includes('--help')) {
        printCliHelp(process.argv[1]);
        return;
    }
    if (args.includes('web')) {
        const { runWebServer } = await import('../../../shared/essentials/web.mjs');
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
