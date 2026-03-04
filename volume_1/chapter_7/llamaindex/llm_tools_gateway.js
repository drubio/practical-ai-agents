/**
 * LLM Tools - LlamaIndex JS with Wikipedia.
 */

import { LlamaIndexLLMManager as Chapter6LlamaIndexManager } from '../../chapter_6/llamaindex/llm_memory_retrieval.js';
import { interactiveCli, getDefaultModel } from '../../chapter_4/utils.js';
import { normalizeResponseText } from '../../chapter_4/stream.js';
import { buildToolsPrompt, runTool } from '../tools.js';

const TOOLS_TEMPLATE = `You are a helpful assistant with access to external tools.

Available tools:
{tools}

Return strict JSON:
{
  "tool_call": null OR {"name": "tool_name", "arguments": {"arg": "value"}},
  "final_answer": "string"
}

Rules:
- If no tool is needed, set tool_call to null.
- If a tool is needed, set tool_call and keep final_answer short.
- Return JSON only.

User topic: {topic}`;

const FOLLOW_UP_TEMPLATE = `You already requested a tool and now have the result.

Original user topic: {topic}
Tool call: {tool_call}
Tool output: {tool_output}

Return strict JSON:
{
  "tool_call": null,
  "final_answer": "final response for the user"
}`;

class LlamaIndexLLMManager extends Chapter6LlamaIndexManager {
    constructor(memoryEnabled = true, retrievalK = 4) {
        super(memoryEnabled, retrievalK);
        this.framework = 'LlamaIndex Tools JS';
    }

    _extractJsonObject(raw) {
        let text = String(raw || '').trim();
        if (text.startsWith('```json')) text = text.slice(7);
        if (text.startsWith('```')) text = text.slice(3);
        if (text.endsWith('```')) text = text.slice(0, -3);
        text = text.trim();

        try {
            return JSON.parse(text);
        } catch {
            const match = text.match(/\{[\s\S]*\}/);
            if (!match) throw new Error('No JSON object found in model response');
            return JSON.parse(match[0]);
        }
    }


    _buildFallbackToolPayload(rawText) {
        const fallbackAnswer = normalizeResponseText(rawText).trim();
        if (!fallbackAnswer) {
            throw new Error('No JSON object found in model response');
        }
        return {
            tool_call: null,
            final_answer: fallbackAnswer,
        };
    }

    _normalizeToolPayload(rawPayload, rawText) {
        if (!rawPayload || typeof rawPayload !== 'object' || Array.isArray(rawPayload)) {
            return this._buildFallbackToolPayload(rawText);
        }

        const toolCall = rawPayload.tool_call;
        const finalAnswer = rawPayload.final_answer;

        if (typeof finalAnswer === 'string' && finalAnswer.trim()) {
            return {
                tool_call: toolCall && typeof toolCall === 'object' ? toolCall : null,
                final_answer: finalAnswer.trim(),
            };
        }

        return this._buildFallbackToolPayload(rawText);
    }

    async _invokeJsonStep(provider, prompt, temperature, maxTokens) {
        const client = this._createClient(provider, temperature, maxTokens);
        const result = await client.chat({ messages: [{ role: 'user', content: prompt }] });
        const text = this._extractText(result);

        try {
            const payload = this._extractJsonObject(text);
            return { payload: this._normalizeToolPayload(payload, text), result };
        } catch {
            return { payload: this._buildFallbackToolPayload(text), result };
        }
    }

    async _buildRetrievalContext(provider, topic, sessionId) {
        const messages = this.retrievalMemoryEnabled
            ? await this._getMemoryMessages(provider, sessionId)
            : [];
        const retrieved = this.retrievalMemoryEnabled
            ? this._selectRetrievedMessages(topic, messages)
            : [];

        const retrievedContext = retrieved.map((item) => `[${item.role}] ${item.content}`).join('\n');
        const retrievalAugmentedTopic = retrievedContext
            ? `Relevant memory snippets:\n${retrievedContext}\n\nCurrent user topic: ${topic}`
            : topic;

        const fullHistoryContext = messages
            .map((msg) => `[${String(msg?.role ?? 'unknown')}] ${String(msg?.content ?? '')}`)
            .join('\n');
        const promptWithoutRetrieval = fullHistoryContext
            ? `${fullHistoryContext}\n\nCurrent user topic: ${topic}`
            : topic;

        const tokensWithRetrieval = this._estimateTokens(retrievalAugmentedTopic);
        const tokensWithoutRetrieval = this._estimateTokens(promptWithoutRetrieval);
        const estimatedSaved = Math.max(0, tokensWithoutRetrieval - tokensWithRetrieval);
        const reductionPercent = tokensWithoutRetrieval > 0
            ? Number(((estimatedSaved / tokensWithoutRetrieval) * 100).toFixed(2))
            : 0;

        return {
            retrievalAugmentedTopic,
            retrievalMetadata: {
                history_messages_available: messages.length,
                retrieved_messages_count: retrieved.length,
                retrieved_messages: retrieved,
                tokens_with_memory_retrieval: tokensWithRetrieval,
                tokens_without_memory_retrieval: tokensWithoutRetrieval,
                estimated_tokens_saved: estimatedSaved,
                estimated_token_reduction_percent: reductionPercent,
            },
        };
    }



    _resolveToolsTemplate(template) {
        const candidate = String(template ?? '').trim();
        if (!candidate || candidate === '{topic}' || !candidate.includes('{tools}')) {
            return TOOLS_TEMPLATE;
        }
        return template;
    }

    _normalizeWikipediaQuery(topic) {
        const text = String(topic || '').trim();
        if (!text) return text;
        return text
            .replace(/^\s*(what\s+is|who\s+is|where\s+is|when\s+did|when\s+was|why\s+is|how\s+is)\s+/i, '')
            .replace(/[?]+$/g, '')
            .trim();
    }

    _shouldForceWikipediaTool(topic, toolCall) {
        if (toolCall && typeof toolCall === 'object' && toolCall.name) return false;
        const text = String(topic || '').toLowerCase();
        if (!text.trim()) return false;

        const creativeSignals = [
            'poem', 'story', 'fiction', 'brainstorm', 'imagine', 'creative writing', 'roleplay', 'joke',
        ];
        if (creativeSignals.some((k) => text.includes(k))) return false;

        const factualSignals = [
            'what is', 'who is', 'when did', 'where is', 'why', 'how', 'define', 'history', 'date', 'science',
        ];
        return factualSignals.some((k) => text.includes(k)) || text.split(/\s+/).length >= 3;
    }

    async askQuestion(topic, provider = null, template = TOOLS_TEMPLATE, maxTokens = 1000, temperature = 0.2, sessionId = 'default') {
        template = this._resolveToolsTemplate(template);
        const resolvedProvider = this._resolveProvider(provider);
        const basePrompt = template.replace('{topic}', topic).replace('{tools}', buildToolsPrompt());

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

        const { retrievalAugmentedTopic, retrievalMetadata } = await this._buildRetrievalContext(resolvedProvider, topic, sessionId);
        const prompt = template.replace('{topic}', retrievalAugmentedTopic).replace('{tools}', buildToolsPrompt());
        const model = getDefaultModel(resolvedProvider);

        try {
            const { payload: firstStep } = await this._invokeJsonStep(resolvedProvider, prompt, temperature, maxTokens);
            const toolCall = firstStep.tool_call;
            let finalAnswer = String(firstStep.final_answer || '').trim();
            let toolOutput = null;

            let effectiveToolCall = toolCall;
            if (this._shouldForceWikipediaTool(topic, effectiveToolCall)) {
                effectiveToolCall = {
                    name: 'get_wikipedia_evidence_pack',
                    arguments: { query: this._normalizeWikipediaQuery(topic) || topic },
                };
            }

            if (effectiveToolCall && typeof effectiveToolCall === 'object' && effectiveToolCall.name) {
                const toolName = String(effectiveToolCall.name);
                const toolArgs = effectiveToolCall.arguments && typeof effectiveToolCall.arguments === 'object' ? effectiveToolCall.arguments : {};
                toolOutput = await runTool(toolName, toolArgs);

                const followUpPrompt = FOLLOW_UP_TEMPLATE
                    .replace('{topic}', topic)
                    .replace('{tool_call}', JSON.stringify(effectiveToolCall))
                    .replace('{tool_output}', String(toolOutput));

                const { payload: secondStep } = await this._invokeJsonStep(resolvedProvider, followUpPrompt, temperature, maxTokens);
                finalAnswer = String(secondStep.final_answer || finalAnswer).trim() || finalAnswer;
            }

            const rawResponse = JSON.stringify({
                tool_call: effectiveToolCall ?? null,
                tool_output: toolOutput,
                final_answer: finalAnswer,
            });

            const responsePayload = {
                tool_call: effectiveToolCall ?? null,
                tool_output: toolOutput,
                final_answer: finalAnswer,
                metadata: {
                    ...this._buildMetadata({
                        provider: resolvedProvider,
                        model,
                        sessionId,
                        temperature,
                        maxTokens,
                    }, rawResponse),
                    retrieval: retrievalMetadata,
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
                model,
                prompt,
                response: responsePayload,
                rawAnswer: finalAnswer,
                temperature,
                maxTokens,
                sessionId,
            };
        } catch (error) {
            return {
                success: false,
                provider: resolvedProvider,
                model,
                prompt,
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
