/**
 * LLM Memory Gateway - LangChain JS with persistent session memory.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { FileChatMessageHistory } from '@langchain/community/stores/message/file';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { RunnableWithMessageHistory } from '@langchain/core/runnables';
import { LangChainLLMManager as Chapter4LangChainManager } from '../../chapter_4/langchain/llm_gateway.js';
import { getDefaultModel, interactiveCli } from '../../chapter_4/utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class LangChainLLMManager extends Chapter4LangChainManager {
    constructor(memoryEnabled = true) {
        super();
        this.framework = 'LangChain+History JS';
        this.memoryEnabled = memoryEnabled;
        this.histories = new Map();
        this.chains = new Map();
    }

    _historyKey(provider, sessionId) {
        return `${provider}::${sessionId}`;
    }

    _sessionFilePath(provider, sessionId) {
        const sessionsDir = path.join(__dirname, 'sessions');
        fs.mkdirSync(sessionsDir, { recursive: true });
        return path.join(sessionsDir, `${provider}__${sessionId}.json`);
    }

    _getHistory(provider, sessionId) {
        const key = this._historyKey(provider, sessionId);
        if (!this.histories.has(key)) {
            const filePath = this._sessionFilePath(provider, sessionId);
            this.histories.set(key, new FileChatMessageHistory({ filePath }));
        }
        return this.histories.get(key);
    }

    _getChain(provider, sessionId, temperature, maxTokens) {
        const key = this._historyKey(provider, sessionId);
        if (!this.chains.has(key)) {
            const client = this._createClient(provider, temperature, maxTokens);
            const prompt = ChatPromptTemplate.fromMessages([
                new MessagesPlaceholder('history'),
                ['human', '{input}'],
            ]);
            const chain = new RunnableWithMessageHistory({
                runnable: prompt.pipe(client),
                getMessageHistory: () => this._getHistory(provider, sessionId),
                inputMessagesKey: 'input',
                historyMessagesKey: 'history',
            });
            this.chains.set(key, chain);
        }
        return this.chains.get(key);
    }

    async askQuestion(topic, provider = null, template = '{topic}', maxTokens = 1000, temperature = 0.7, sessionId = 'default') {
        if (!this.memoryEnabled) {
            return super.askQuestion(topic, provider, template, maxTokens, temperature);
        }

        const prompt = template.replace('{topic}', topic);
        const resolvedProvider = this._resolveProvider(provider);

        if (!resolvedProvider) {
            return { success: false, error: 'No providers available', provider: 'none', model: 'none', prompt, response: null };
        }

        const model = getDefaultModel(resolvedProvider);

        try {
            const chain = this._getChain(resolvedProvider, sessionId, temperature, maxTokens);
            const result = await chain.invoke({ input: prompt }, { configurable: { sessionId } });
            const responseText = this._extractText(resolvedProvider, result);

            return { success: true, provider: resolvedProvider, model, prompt, response: responseText, temperature, maxTokens, sessionId };
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

    getHistory(provider, sessionId = 'default') {
        const turns = this._getHistory(provider, sessionId).messages.map((message) => ({
            role: message._getType?.() ?? message.getType?.() ?? message.type,
            content: message.content,
        }));
        return { provider, sessionId, turns, count: turns.length };
    }

    resetMemory(provider = null, sessionId = null) {
        const removedSessions = [];
        const sessionsDir = path.join(__dirname, 'sessions');

        if (provider && sessionId) {
            const key = this._historyKey(provider, sessionId);
            this.histories.delete(key);
            this.chains.delete(key);
            removedSessions.push([provider, sessionId]);
        } else if (provider) {
            for (const key of Array.from(this.histories.keys())) {
                const [p, s] = key.split('::');
                if (p === provider) {
                    this.histories.delete(key);
                    this.chains.delete(key);
                    removedSessions.push([p, s]);
                }
            }
        } else if (sessionId) {
            for (const key of Array.from(this.histories.keys())) {
                const [p, s] = key.split('::');
                if (s === sessionId) {
                    this.histories.delete(key);
                    this.chains.delete(key);
                    removedSessions.push([p, s]);
                }
            }
        } else {
            this.histories.clear();
            this.chains.clear();
            removedSessions.push('ALL');
        }

        if (removedSessions.length === 1 && removedSessions[0] === 'ALL') {
            if (fs.existsSync(sessionsDir)) {
                for (const file of fs.readdirSync(sessionsDir)) {
                    if (file.endsWith('.json')) fs.unlinkSync(path.join(sessionsDir, file));
                }
            }
        } else {
            for (const [p, s] of removedSessions.filter(Array.isArray)) {
                const file = this._sessionFilePath(p, s);
                if (fs.existsSync(file)) fs.unlinkSync(file);
            }
        }

        return { status: 'cleared', removedSessions };
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
