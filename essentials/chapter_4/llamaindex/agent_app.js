/**
 * "LLM application to chat with multiple LLMs - LlamaIndex JavaScript framework implementation
 */

import { createLlamaIndexModel } from '../../../shared/utils.mjs';

import {
    BaseLLMManager,
    interactiveCli,
} from '../../../shared/essentials/utils.mjs';

class LlamaIndexLLMManager extends BaseLLMManager {
    constructor() {
        super('LlamaIndex JS');
    }

    async _testProvider(provider) {
        await this._createModel(this.providerModelIdentifier(provider), 0.7, 1000);
    }

    _createModel(selectedModel, temperature, maxTokens) {
        return createLlamaIndexModel(selectedModel, {
            temperature,
            maxTokens,
        });
    }

    _resolveProvider(provider) {
        return this.resolveModelIdentifier(provider);
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
        const modelConfig = this.resolveModelConfig(provider);

        if (!modelConfig) {
            return {
                success: false,
                error: 'No providers available',
                provider: 'none',
                model: 'none',
                prompt,
                response: null,
            };
        }

        try {
            const model = this._createModel(modelConfig.name, temperature, maxTokens);
            const result = await model.chat({
                messages: [{ role: 'user', content: prompt }],
            });

            return {
                success: true,
                provider: modelConfig.provider,
                model: modelConfig.model,
                modelIdentifier: modelConfig.name,
                prompt,
                response: this._extractText(result),
                temperature,
                maxTokens,
            };
        } catch (error) {
            return {
                success: false,
                provider: modelConfig.provider,
                model: modelConfig.model,
                modelIdentifier: modelConfig.name,
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
