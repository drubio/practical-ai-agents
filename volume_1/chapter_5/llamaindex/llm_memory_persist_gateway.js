/**
 * LLM Memory Gateway - LlamaIndex JS with persistent session memory.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { ChatMemoryBuffer, SimpleChatEngine, SimpleChatStore } from 'llamaindex';
import { LlamaIndexLLMManager as Chapter4LlamaIndexManager } from '../../chapter_4/llamaindex/llm_gateway.js';
import { getDefaultModel, interactiveCli } from '../../chapter_4/utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class LlamaIndexLLMManager extends Chapter4LlamaIndexManager {
    constructor(memoryEnabled = true) {
        super();
        this.framework = 'LlamaIndex+History JS';
        this.memoryEnabled = memoryEnabled;
        this.memories = new Map();
        this.chatEngines = new Map();
        this.chatStores = new Map();
    }

    _historyKey(provider, sessionId) {
        return `${provider}::${sessionId}`;
    }

    _sessionFilePath(provider, sessionId) {
        const sessionsDir = path.join(__dirname, 'sessions');
        fs.mkdirSync(sessionsDir, { recursive: true });
        return path.join(sessionsDir, `${provider}__${sessionId}.json`);
    }

    _sessionStoreKey(provider, sessionId) {
        return `${provider}__${sessionId}`;
    }

    _getChatStore(provider, sessionId) {
        const key = this._historyKey(provider, sessionId);
        if (!this.chatStores.has(key)) {
            const filePath = this._sessionFilePath(provider, sessionId);
            const store = fs.existsSync(filePath)
                ? SimpleChatStore.fromPersistPath(filePath)
                : new SimpleChatStore();
            this.chatStores.set(key, store);
        }
        return this.chatStores.get(key);
    }

    _getMemory(provider, sessionId) {
        const key = this._historyKey(provider, sessionId);
        if (!this.memories.has(key)) {
            const chatStore = this._getChatStore(provider, sessionId);
            const memory = ChatMemoryBuffer.fromDefaults({
                chatStore,
                chatStoreKey: this._sessionStoreKey(provider, sessionId),
            });
            this.memories.set(key, memory);
        }
        return this.memories.get(key);
    }

    _persistMemory(provider, sessionId) {
        const chatStore = this._getChatStore(provider, sessionId);
        chatStore.persist(this._sessionFilePath(provider, sessionId));
    }

    _getChatEngine(provider, sessionId, temperature, maxTokens) {
        const key = this._historyKey(provider, sessionId);
        if (!this.chatEngines.has(key)) {
            const client = this._createClient(provider, temperature, maxTokens);
            const memory = this._getMemory(provider, sessionId);
            const engine = SimpleChatEngine.fromDefaults({ llm: client, memory });
            this.chatEngines.set(key, engine);
        }
        return this.chatEngines.get(key);
    }

    _memoryMessages(memory) {
        if (typeof memory.getAll === 'function') return memory.getAll();
        if (typeof memory.getMessages === 'function') return memory.getMessages();
        return Array.isArray(memory.chatHistory) ? memory.chatHistory : [];
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
            const chatEngine = this._getChatEngine(resolvedProvider, sessionId, temperature, maxTokens);
            const response = await chatEngine.chat(prompt);
            const responseText = response?.response ?? this._extractText(response);
            this._persistMemory(resolvedProvider, sessionId);

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
        const turns = this._memoryMessages(this._getMemory(provider, sessionId)).map((message) => ({
            role: message.role ?? message?.message?.role,
            content: message.content ?? message?.message?.content,
        }));
        return { provider, sessionId, turns, count: turns.length };
    }

    resetMemory(provider = null, sessionId = null) {
        const removedSessions = [];
        const sessionsDir = path.join(__dirname, 'sessions');

        if (provider && sessionId) {
            const key = this._historyKey(provider, sessionId);
            this.memories.delete(key);
            this.chatEngines.delete(key);
            this.chatStores.delete(key);
            removedSessions.push([provider, sessionId]);
        } else if (provider) {
            for (const key of Array.from(this.memories.keys())) {
                const [p, s] = key.split('::');
                if (p === provider) {
                    this.memories.delete(key);
                    this.chatEngines.delete(key);
                    this.chatStores.delete(key);
                    removedSessions.push([p, s]);
                }
            }
        } else if (sessionId) {
            for (const key of Array.from(this.memories.keys())) {
                const [p, s] = key.split('::');
                if (s === sessionId) {
                    this.memories.delete(key);
                    this.chatEngines.delete(key);
                    this.chatStores.delete(key);
                    removedSessions.push([p, s]);
                }
            }
        } else {
            this.memories.clear();
            this.chatEngines.clear();
            this.chatStores.clear();
            removedSessions.push('ALL');
        }

        if (removedSessions.length === 1 && removedSessions[0] === 'ALL') {
            if (fs.existsSync(sessionsDir)) {
                for (const fileName of fs.readdirSync(sessionsDir)) {
                    if (fileName.endsWith('.json')) fs.unlinkSync(path.join(sessionsDir, fileName));
                }
            }
        } else {
            for (const [p, s] of removedSessions.filter(Array.isArray)) {
                const filePath = this._sessionFilePath(p, s);
                if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
            }
        }

        return { status: 'cleared', removedSessions };
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
