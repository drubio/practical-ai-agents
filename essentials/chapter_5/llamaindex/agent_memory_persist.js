/**
 * Agent Memory Gateway - LlamaIndex JS with persistent session memory.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { SimpleChatEngine } from '@llamaindex/core/chat-engine';
import { Memory } from '@llamaindex/core/memory';
import { SimpleChatStore } from '@llamaindex/core/storage/chat-store';
import { LlamaIndexLLMManager as Chapter4LlamaIndexManager } from '../../chapter_4/llamaindex/agent_app.js';
import { interactiveCli } from '../../../shared/essentials/utils.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function buildPythonStorePayload(storeKey, messages) {
    return {
        store: {
            [storeKey]: messages,
        },
        class_name: 'SimpleChatStore',
    };
}

function extractMessageText(message) {
    if (typeof message?.content === 'string') return message.content;
    if (Array.isArray(message?.content)) {
        return message.content
            .filter((item) => item?.type === 'text' && typeof item.text === 'string')
            .map((item) => item.text)
            .join('\n\n');
    }
    if (Array.isArray(message?.blocks)) {
        return message.blocks
            .filter((block) => block?.block_type === 'text' && typeof block.text === 'string')
            .map((block) => block.text)
            .join('\n\n');
    }
    return '';
}

function normalizePersistedMessage(message) {
    return {
        ...message,
        content: extractMessageText(message),
    };
}

function normalizePersistedMessages(messages) {
    return messages.map((message) => normalizePersistedMessage(message));
}

class LlamaIndexLLMManager extends Chapter4LlamaIndexManager {
    constructor(memoryEnabled = true) {
        super();
        this.framework = 'LlamaIndex Memory+Persistence JS';
        this.memoryEnabled = memoryEnabled;
        this.memories = new Map();
        this.chatEngines = new Map();
        this.chatStores = new Map();
    }

    _sessionKey(provider, sessionId) {
        return `${provider}::${sessionId}`;
    }

    _sessionStoreKey(sessionId) {
        return sessionId;
    }

    _sessionFilePath(sessionId) {
        const sessionsDir = path.join(__dirname, 'sessions');
        fs.mkdirSync(sessionsDir, { recursive: true });
        return path.join(sessionsDir, `${sessionId}.json`);
    }

    _getChatStore(sessionId) {
        if (this.chatStores.has(sessionId)) {
            return this.chatStores.get(sessionId);
        }

        const chatStore = new SimpleChatStore();
        const storeKey = this._sessionStoreKey(sessionId);
        const filePath = this._sessionFilePath(sessionId);

        if (fs.existsSync(filePath)) {
            const payload = JSON.parse(fs.readFileSync(filePath, 'utf8'));
            const messages = payload?.store?.[storeKey];
            chatStore.setMessages(storeKey, Array.isArray(messages) ? normalizePersistedMessages(messages) : []);
        }

        this.chatStores.set(sessionId, chatStore);
        return chatStore;
    }

    _getMemory(sessionId) {
        if (!this.memories.has(sessionId)) {
            const storeKey = this._sessionStoreKey(sessionId);
            const chatHistory = [...this._getChatStore(sessionId).getMessages(storeKey)];

            // Python parity:
            // Memory.from_defaults(
            //     session_id=store_key,
            //     chat_history=list(chat_store.get_messages(store_key)),
            // )
            // The JS Memory API does not expose `from_defaults` / `session_id`,
            // so we instantiate with the same chat history payload.
            this.memories.set(sessionId, new Memory(chatHistory));
        }
        return this.memories.get(sessionId);
    }

    async _appendToMemory(sessionId, role, content) {
        const memory = this._getMemory(sessionId);
        if (typeof memory.add === 'function') {
            await memory.add({ role, content });
        } else if (typeof memory.put === 'function') {
            await memory.put({ role, content });
        }
    }

    async _persistMemory(sessionId) {
        const storeKey = this._sessionStoreKey(sessionId);
        const messages = await this._getMemory(sessionId).get({ type: 'llamaindex' });
        const chatStore = this._getChatStore(sessionId);
        chatStore.setMessages(storeKey, messages);
        const filePath = this._sessionFilePath(sessionId);
        fs.writeFileSync(filePath, JSON.stringify(buildPythonStorePayload(storeKey, messages)), 'utf8');
    }

    _getChatEngine(provider, sessionId, temperature, maxTokens) {
        const key = this._sessionKey(provider, sessionId);
        if (!this.chatEngines.has(key)) {
            const model = this._createModel(provider, temperature, maxTokens);
            const memory = this._getMemory(sessionId);
            this.chatEngines.set(key, new SimpleChatEngine({ llm: model, memory }));
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

        const modelConfig = this.resolveModelConfig(resolvedProvider);

        try {
            const chatEngine = this._getChatEngine(resolvedProvider, sessionId, temperature, maxTokens);
            const result = await chatEngine.chat({ message: prompt });
            const responseText = result?.response ?? this._extractText(result);
            await this._persistMemory(sessionId);

            return { success: true, provider: modelConfig.provider, model: modelConfig.model, modelIdentifier: modelConfig.name, prompt, response: responseText, temperature, maxTokens, sessionId };
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
                sessionId,
            };
        }
    }

    getHistory(provider, sessionId = 'default') {
        const turns = this._getChatStore(sessionId)
            .getMessages(this._sessionStoreKey(sessionId))
            .map((message) => ({ role: String(message.role), content: message.content }));
        return { provider, sessionId, turns, count: turns.length };
    }

    resetMemory(provider = null, sessionId = null) {
        const removedSessions = [];
        const sessionsDir = path.join(__dirname, 'sessions');

        if (sessionId) {
            this.memories.delete(sessionId);
            this.chatStores.delete(sessionId);
            for (const key of Array.from(this.chatEngines.keys())) {
                if (key.endsWith(`::${sessionId}`)) this.chatEngines.delete(key);
            }
            removedSessions.push(sessionId);
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
            for (const session of removedSessions) {
                const filePath = this._sessionFilePath(session);
                if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
            }
        }

        return { status: 'cleared', removedSessions };
    }
}

async function main() {
    const args = process.argv.slice(2);
    if (args.length > 0 && args[0] === 'web') {
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
