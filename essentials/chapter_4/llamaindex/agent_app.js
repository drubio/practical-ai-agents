/**
 * LLM application to chat with multiple LLMs - LlamaIndex JavaScript framework implementation.
 */

import {
    BaseLLMManager,
    createLlamaIndexModel,
    interactiveCli,
} from '../../../shared/utils.mjs';

class LlamaIndexLLMManager extends BaseLLMManager {
    constructor(stream = false) {
        super('LlamaIndex JS');
        this.stream = stream;
        this.printsOwnOutput = stream;
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

    _extractStreamDelta(chunk) {
        for (const attr of ['delta', 'contentDelta', 'content_delta']) {
            const value = chunk?.[attr];
            if (typeof value === 'string' && value) return value;
        }
        const content = chunk?.message?.content;
        if (typeof content === 'string') return content;
        return chunk ? String(chunk) : '';
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

    async _streamModel(model, messages) {
        const parts = [];
        const stream = await model.streamChat({ messages });
        for await (const chunk of stream) {
            const delta = this._extractStreamDelta(chunk);
            if (delta) {
                process.stdout.write(delta);
                parts.push(delta);
            }
        }
        console.log();
        return parts.join('');
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
            const messages = [{ role: 'user', content: prompt }];
            const response = this.stream
                ? await this._streamModel(model, messages)
                : this._extractText(await model.chat({ messages }));

            return {
                success: true,
                provider: modelConfig.provider,
                model: modelConfig.model,
                modelIdentifier: modelConfig.name,
                prompt,
                response,
                temperature,
                maxTokens,
                stream: this.stream,
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
                stream: this.stream,
            };
        }
    }
}

async function main() {
    const args = process.argv.slice(2);
    const stream = args.includes('--stream');

    if (args.includes('web')) {
        try {
            const { runWebServer } = await import('../../../shared/essentials/web.mjs');
            await runWebServer(() => new LlamaIndexLLMManager(stream));
        } catch (error) {
            console.error('Error: shared web API not found or Express not installed.');
            console.error('Install Express: npm install express cors');
            process.exit(1);
        }
    } else {
        const manager = new LlamaIndexLLMManager(stream);
        await manager._checkProviders();
        await interactiveCli(manager);
    }
}

export { LlamaIndexLLMManager };

if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}
