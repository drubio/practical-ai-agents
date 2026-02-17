/**
 * LLM Tools Gateway - LlamaIndex JS (Chapter 7).
 *
 * Chapter flow: 4 (providers) -> 5 (memory/persistence) ->
 * 6 (retrieval memory + structured output) -> 7 (tools).
 */

import { LlamaIndexLLMManager as Chapter6LlamaIndexManager } from '../../chapter_6/llamaindex/llm_memory_retrieval_gateway.js';
import { interactiveCli, getDefaultModel } from '../../chapter_4/utils.js';
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
        this.framework = 'LlamaIndex+Memory+Retrieval+Tools JS';
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

    async _invokeJsonStep(provider, prompt, temperature, maxTokens) {
        const client = this._createClient(provider, temperature, maxTokens);
        const result = await client.chat({ messages: [{ role: 'user', content: prompt }] });
        const text = this._extractText(result);
        return this._extractJsonObject(text);
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

    async askQuestion(topic, provider = null, template = TOOLS_TEMPLATE, maxTokens = 1000, temperature = 0.2, sessionId = 'default') {
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
            const firstStep = await this._invokeJsonStep(resolvedProvider, prompt, temperature, maxTokens);
            const toolCall = firstStep.tool_call;
            let finalAnswer = String(firstStep.final_answer || '').trim();
            let toolOutput = null;

            if (toolCall && typeof toolCall === 'object' && toolCall.name) {
                const toolName = String(toolCall.name);
                const toolArgs = toolCall.arguments && typeof toolCall.arguments === 'object' ? toolCall.arguments : {};
                toolOutput = runTool(toolName, toolArgs);

                const followUpPrompt = FOLLOW_UP_TEMPLATE
                    .replace('{topic}', topic)
                    .replace('{tool_call}', JSON.stringify(toolCall))
                    .replace('{tool_output}', String(toolOutput));

                const secondStep = await this._invokeJsonStep(resolvedProvider, followUpPrompt, temperature, maxTokens);
                finalAnswer = String(secondStep.final_answer || finalAnswer).trim() || finalAnswer;
            }

            const rawResponse = JSON.stringify({
                tool_call: toolCall ?? null,
                tool_output: toolOutput,
                final_answer: finalAnswer,
            });

            const responsePayload = {
                tool_call: toolCall ?? null,
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
