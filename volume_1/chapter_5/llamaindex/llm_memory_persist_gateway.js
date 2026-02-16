/**
 * LLM Memory Gateway - LlamaIndex JS with persistent session memory.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { SimpleChatEngine } from '../../chapter_4/node_modules/@llamaindex/core/chat-engine/dist/index.js';
import { Memory } from '../../chapter_4/node_modules/@llamaindex/core/memory/dist/index.js';
import { SimpleChatStore } from '../../chapter_4/node_modules/@llamaindex/core/storage/chat-store/dist/index.js';
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

    _sessionKey(provider, sessionId) {
        return `${provider}::${sessionId}`;
    }

    _sessionStoreKey(provider, sessionId) {
        return `${provider}__${sessionId}`;
    }

    _sessionFilePath(provider, sessionId) {
        const sessionsDir = path.join(__dirname, 'sessions');
        fs.mkdirSync(sessionsDir, { recursive: true });
        return path.join(sessionsDir, `${provider}__${sessionId}.json`);
    }

    _getChatStore(provider, sessionId) {
        const key = this._sessionKey(provider, sessionId);
        if (this.chatStores.has(key)) {
            return this.chatStores.get(key);
        }

        const chatStore = new SimpleChatStore();
        const storeKey = this._sessionStoreKey(provider, sessionId);
        const filePath = this._sessionFilePath(provider, sessionId);

        if (fs.existsSync(filePath)) {
            try {
                const payload = JSON.parse(fs.readFileSync(filePath, 'utf8'));
                if (Array.isArray(payload?.messages)) {
                    chatStore.setMessages(storeKey, payload.messages);
                } else if (Array.isArray(payload?.turns)) {
                    chatStore.setMessages(storeKey, payload.turns);
                } else if (Array.isArray(payload)) {
                    chatStore.setMessages(storeKey, payload);
                }
            } catch {
                chatStore.setMessages(storeKey, []);
            }
        }

        this.chatStores.set(key, chatStore);
        return chatStore;
    }

    _getMemory(provider, sessionId) {
        const key = this._sessionKey(provider, sessionId);
        if (!this.memories.has(key)) {
            const storeKey = this._sessionStoreKey(provider, sessionId);
            const chatHistory = [...this._getChatStore(provider, sessionId).getMessages(storeKey)];

            // Python parity:
            // Memory.from_defaults(
            //     session_id=store_key,
            //     chat_history=list(chat_store.get_messages(store_key)),
            // )
            // The JS Memory API does not expose `from_defaults` / `session_id`,
            // so we instantiate with the same chat history payload.
            this.memories.set(key, new Memory(chatHistory));
        }
        return this.memories.get(key);
    }

    async _persistMemory(provider, sessionId) {
        const storeKey = this._sessionStoreKey(provider, sessionId);
        const messages = await this._getMemory(provider, sessionId).get({ type: 'llamaindex' });
        this._getChatStore(provider, sessionId).setMessages(storeKey, messages);
        const filePath = this._sessionFilePath(provider, sessionId);
        fs.writeFileSync(filePath, JSON.stringify({ messages }, null, 2), 'utf8');
    }

    _getChatEngine(provider, sessionId, temperature, maxTokens) {
        const key = this._sessionKey(provider, sessionId);
        if (!this.chatEngines.has(key)) {
            const client = this._createClient(provider, temperature, maxTokens);
            const memory = this._getMemory(provider, sessionId);
            this.chatEngines.set(key, new SimpleChatEngine({ llm: client, memory }));
        }
        return this.chatEngines.get(key);
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
            const result = await chatEngine.chat({ message: prompt });
            const responseText = result?.response ?? this._extractText(result);
            await this._persistMemory(resolvedProvider, sessionId);

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
        const turns = this._getChatStore(provider, sessionId)
            .getMessages(this._sessionStoreKey(provider, sessionId))
            .map((message) => ({ role: String(message.role), content: message.content }));
        return { provider, sessionId, turns, count: turns.length };
    }

    resetMemory(provider = null, sessionId = null) {
        const removedSessions = [];
        const sessionsDir = path.join(__dirname, 'sessions');

        if (provider && sessionId) {
            const key = this._sessionKey(provider, sessionId);
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
