/**
 * "LLM application to chat with multiple LLMs - LlamaIndex JavaScript framework implementation
 */

import { Anthropic } from '@llamaindex/anthropic';
import { OpenAI } from '@llamaindex/openai';
import { Gemini } from '@llamaindex/google';

import {
    getApiKey,
    getDefaultModel,
    BaseLLMManager,
    interactiveCli,
} from '../../../shared/essentials/utils.mjs';

const GOOGLE_GEMINI_FALLBACK_CONTEXT_WINDOW = 1_000_000;
const GOOGLE_GEMINI_FALLBACK_MODELS = new Set([
    'gemini-3-flash-preview',
]);

// This is patch to support newer Gemini version (e.g. 3.0)
// since the LlamaIndex package is no longer being updated beyond 2.5 models
class CompatibleGemini extends Gemini {
    get metadata() {
        try {
            return super.metadata;
        } catch (error) {
            if (!GOOGLE_GEMINI_FALLBACK_MODELS.has(this.model)) {
                throw error;
            }

            return {
                model: this.model,
                temperature: this.temperature,
                topP: this.topP,
                maxTokens: this.maxTokens,
                contextWindow: GOOGLE_GEMINI_FALLBACK_CONTEXT_WINDOW,
                tokenizer: undefined,
                structuredOutput: false,
                safetySettings: this.safetySettings,
            };
        }
    }
}

class LlamaIndexLLMManager extends BaseLLMManager {
    constructor() {
        super('LlamaIndex JS');
    }

    async _testProvider(provider) {
        await this._createClient(provider, 0.7, 1000);
    }

    _createClient(provider, temperature, maxTokens) {
        if (provider === 'anthropic') {
            return new Anthropic({
                apiKey: getApiKey(provider),
                model: getDefaultModel(provider),
                temperature,
                maxTokens,
            });
        }
        if (provider === 'openai') {
            return new OpenAI({
                apiKey: getApiKey(provider),
                model: getDefaultModel(provider),
                temperature,
                maxCompletionTokens: maxTokens,
            });
        }
	// Using patch CompatibleGemini vs. native Gemini llamaindex (see above)
        if (provider === 'google') {
            return new CompatibleGemini({
                apiKey: getApiKey(provider),
                model: getDefaultModel(provider),
                temperature,
                maxTokens,
            });
        }
        if (provider === 'xai') {
            return new OpenAI({
                apiKey: getApiKey(provider),
                baseURL: 'https://api.x.ai/v1',
                model: getDefaultModel(provider),
                temperature,
                maxCompletionTokens: maxTokens,
            });
        }
        if (provider === 'deepseek') {
            return new OpenAI({
                apiKey: getApiKey(provider),
                baseURL: 'https://api.deepseek.com',
                model: getDefaultModel(provider),
                temperature,
                maxCompletionTokens: maxTokens,
            });
        }	
        throw new Error(`Unsupported provider: ${provider}`);
    }

    _resolveProvider(provider) {
        const available = this.getAvailableProviders();
        if (provider && available.includes(provider)) {
            return provider;
        }
        return available.length > 0 ? available[0] : null;
    }

    _extractText(result) {
        const content = result?.message?.content;
        if (typeof content === 'string') {
            return content;
        }
        if (Array.isArray(content)) {
            return content
                .filter((block) => block?.type === 'text' && typeof block?.text === 'string')
                .map((block) => block.text)
                .join('\n');
        }
        return String(content ?? result?.message ?? result ?? '');
    }

    async askQuestion(topic, provider = null, template = '{topic}', maxTokens = 1000, temperature = 0.7) {
        const prompt = template.replace('{topic}', topic);
        const resolvedProvider = this._resolveProvider(provider);

        if (!resolvedProvider) {
            return {
                success: false,
                error: 'No providers available',
                provider: 'none',
                model: 'none',
                prompt,
                response: null,
            };
        }

        const model = getDefaultModel(resolvedProvider);

        try {
            const client = this._createClient(resolvedProvider, temperature, maxTokens);
            const result = await client.chat({
                messages: [{ role: 'user', content: prompt }],
            });

            return {
                success: true,
                provider: resolvedProvider,
                model,
                prompt,
                response: this._extractText(result),
                temperature,
                maxTokens,
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
            };
        }
    }
}

async function main() {
    const args = process.argv.slice(2);

    if (args.length > 0 && args[0] === 'web') {
        try {
            const { runWebServer } = await import('../../../shared/essentials/web.mjs');
            await runWebServer(LlamaIndexLLMManager);
        } catch (error) {
            console.error('Error: shared web API not found or Express not installed.');
            console.error('Install Express: npm install express cors');
            process.exit(1);
        }
    } else {
        const manager = new LlamaIndexLLMManager();
        await manager._checkProviders();
        await interactiveCli(manager);
    }
}

export { LlamaIndexLLMManager };

if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}
