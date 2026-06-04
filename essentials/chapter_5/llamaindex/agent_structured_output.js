/**
 * Agent Memory Structured Gateway - LlamaIndex JS with structured JSON responses.
 */

import { LlamaIndexLLMManager as Chapter5LlamaIndexManager } from './agent_memory_persist.js';
import { interactiveCli, parseStructuredJsonResponse, printCliHelp } from '../../../shared/essentials/utils.mjs';

export const STRUCTURED_TEMPLATE = `Given the topic below, provide:

1. A direct factual answer (if possible)
2. A summary of what the question is about
3. Relevant keywords
4. A distilled answer (short phrase or value-only form of the answer)

Respond in the following JSON format:
{
  "answer": "...",
  "summary": "...",
  "keywords": ["...", "..."],
  "distilled": "...",
  "metadata": {
    "confidence": "high|medium|low",
    "notes": "optional extra context"
  }
}

Topic: {topic}`;

class LlamaIndexLLMManager extends Chapter5LlamaIndexManager {
    constructor(memoryEnabled = true) {
        super(memoryEnabled);
        this.framework = 'LlamaIndex Structured Output JS';
    }

    _extractObjectFromText(rawResponse, key) {
        const marker = `${key}=`;
        const markerIndex = rawResponse.indexOf(marker);
        if (markerIndex === -1) return null;

        const start = rawResponse.indexOf('{', markerIndex);
        if (start === -1) return null;

        let depth = 0;
        let inString = false;
        let quote = '';
        let escape = false;

        for (let i = start; i < rawResponse.length; i += 1) {
            const ch = rawResponse[i];
            if (inString) {
                if (escape) {
                    escape = false;
                } else if (ch === '\\') {
                    escape = true;
                } else if (ch === quote) {
                    inString = false;
                }
                continue;
            }

            if (ch === '"' || ch === "'") {
                inString = true;
                quote = ch;
                continue;
            }

            if (ch === '{') {
                depth += 1;
            } else if (ch === '}') {
                depth -= 1;
                if (depth === 0) {
                    const block = rawResponse.slice(start, i + 1);
                    try {
                        return JSON.parse(block.replace(/'/g, '"'));
                    } catch {
                        return null;
                    }
                }
            }
        }

        return null;
    }

    _buildMetadata(result, rawResponse) {
        const responseMetadata = result.response_metadata ?? this._extractObjectFromText(rawResponse, 'response_metadata');
        const usageMetadata = result.usage_metadata ?? this._extractObjectFromText(rawResponse, 'usage_metadata');
        const tokenUsageSource = responseMetadata?.token_usage;
        const tokenUsage = tokenUsageSource
            ? {
                completion_tokens: tokenUsageSource.completion_tokens,
                prompt_tokens: tokenUsageSource.prompt_tokens,
                total_tokens: tokenUsageSource.total_tokens,
            }
            : (usageMetadata
                ? {
                    completion_tokens: usageMetadata.output_tokens,
                    prompt_tokens: usageMetadata.input_tokens,
                    total_tokens: usageMetadata.total_tokens,
                }
                : null);

        const metadata = {
            provider: result.provider,
            model: result.model,
            framework: this.framework,
            session_id: result.sessionId,
            temperature: result.temperature,
            max_tokens: result.maxTokens,
            raw_response_chars: rawResponse.length,
            token_usage: tokenUsage ? Object.fromEntries(Object.entries(tokenUsage).filter(([, value]) => value != null)) : null,
        };

        return Object.fromEntries(Object.entries(metadata).filter(([, value]) => value != null));
    }

    async askQuestion(topic, provider = null, template = STRUCTURED_TEMPLATE, maxTokens = 1000, temperature = 0.7, sessionId = 'default') {
        const effectiveTemplate = template === '{topic}' ? STRUCTURED_TEMPLATE : template;
        const result = await super.askQuestion(topic, provider, effectiveTemplate, maxTokens, temperature, sessionId);

        if (!result.success) {
            return result;
        }

        const rawResponse = typeof result.response === 'string' ? result.response : String(result.response ?? '');
        try {
            const parsed = parseStructuredJsonResponse(rawResponse);
            parsed.metadata = {
                ...(parsed.metadata || {}),
                ...this._buildMetadata(result, rawResponse),
            };
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
    if (args.includes('-h') || args.includes('--help')) {
        printCliHelp(process.argv[1]);
        return;
    }
    if (args.includes('web')) {
        const { runWebServer } = await import('../../../shared/essentials/web.mjs');
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
