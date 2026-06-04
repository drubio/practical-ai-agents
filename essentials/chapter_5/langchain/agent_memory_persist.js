/**
 * LLM Memory Gateway - LangChain JS with persistent session memory.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { FileSystemChatMessageHistory } from '@langchain/community/stores/message/file_system';
import { mapChatMessagesToStoredMessages } from '@langchain/core/messages';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { RunnableWithMessageHistory } from '@langchain/core/runnables';
import { LangChainLLMManager as Chapter4LangChainManager } from '../../chapter_4/langchain/agent_app.js';
import { interactiveCli, printCliHelp } from '../../../shared/essentials/utils.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function patchFileSystemChatMessageHistory(history) {
    history.loadStore = async function loadSharedStore() {
        try {
            const raw = await fs.promises.readFile(this.filePath, 'utf8');
            if (!raw.trim()) return {};

            const payload = JSON.parse(raw);
            if (Array.isArray(payload)) {
                return {
                    [this.userId ?? '']: {
                        [this.sessionId]: { messages: payload },
                    },
                };
            }
            if (Array.isArray(payload?.messages)) {
                return {
                    [this.userId ?? '']: {
                        [this.sessionId]: { messages: payload.messages },
                    },
                };
            }

            return payload;
        } catch (error) {
            if (error.code === 'ENOENT') return {};
            throw new Error(`Error loading FileSystemChatMessageHistory store: ${error}`);
        }
    };

    history.saveStore = async function saveSharedStore() {
        try {
            const messages = await this.getMessages();
            const storedMessages = mapChatMessagesToStoredMessages(messages);
            await fs.promises.mkdir(path.dirname(this.filePath), { recursive: true });
            await fs.promises.writeFile(this.filePath, JSON.stringify(storedMessages));
        } catch (error) {
            throw new Error(`Error saving FileSystemChatMessageHistory store: ${error}`);
        }
    };

    return history;
}

class LangChainLLMManager extends Chapter4LangChainManager {
    constructor(memoryEnabled = true) {
        super();
        this.framework = 'LangChain Memory+Persistence JS';
        this.memoryEnabled = memoryEnabled;
        this.histories = new Map();
        this.chains = new Map();
    }

    _historyKey(provider, sessionId) {
        return `${provider}::${sessionId}`;
    }

    _sessionFilePath(sessionId) {
        const sessionsDir = path.join(__dirname, 'sessions');
        fs.mkdirSync(sessionsDir, { recursive: true });
        return path.join(sessionsDir, `${sessionId}.json`);
    }

    _getHistory(sessionId) {
        if (!this.histories.has(sessionId)) {
            const filePath = this._sessionFilePath(sessionId);
            this.histories.set(
                sessionId,
                patchFileSystemChatMessageHistory(new FileSystemChatMessageHistory({ sessionId, filePath })),
            );
        }
        return this.histories.get(sessionId);
    }

    _getChain(provider, sessionId, temperature, maxTokens) {
        const key = this._historyKey(provider, sessionId);
        if (!this.chains.has(key)) {
            const model = this._createModel(provider, temperature, maxTokens);
            const prompt = ChatPromptTemplate.fromMessages([
                new MessagesPlaceholder('history'),
                ['human', '{input}'],
            ]);
            const chain = new RunnableWithMessageHistory({
                runnable: prompt.pipe(model),
                getMessageHistory: () => this._getHistory(sessionId),
                inputMessagesKey: 'input',
                historyMessagesKey: 'history',
            });
            this.chains.set(key, chain);
        }
        return this.chains.get(key);
    }

    _extractTokenUsage(responseMetadata, usageMetadata) {
        if (responseMetadata && typeof responseMetadata === 'object') {
            const source = (responseMetadata.token_usage && typeof responseMetadata.token_usage === 'object')
                ? responseMetadata.token_usage
                : responseMetadata;
            const compact = {
                completion_tokens: source.completion_tokens,
                prompt_tokens: source.prompt_tokens,
                total_tokens: source.total_tokens,
            };
            const cleaned = Object.fromEntries(Object.entries(compact).filter(([, value]) => value != null));
            if (Object.keys(cleaned).length > 0) return cleaned;
        }

        if (usageMetadata && typeof usageMetadata === 'object') {
            const compact = {
                completion_tokens: usageMetadata.output_tokens ?? usageMetadata.completion_tokens,
                prompt_tokens: usageMetadata.input_tokens ?? usageMetadata.prompt_tokens,
                total_tokens: usageMetadata.total_tokens,
            };
            const cleaned = Object.fromEntries(Object.entries(compact).filter(([, value]) => value != null));
            if (Object.keys(cleaned).length > 0) return cleaned;
        }

        return null;
    }

    async askQuestion(topic, provider = null, template = '{topic}', maxTokens = 1000, temperature = 0.7, sessionId = 'default') {
        if (!this.memoryEnabled) {
            return super.askQuestion(topic, provider, template, maxTokens, temperature);
        }

        const prompt = template.replace('{topic}', topic);
        const resolvedProvider = this.resolveModelIdentifier(provider);

        if (!resolvedProvider) {
            return { success: false, error: 'No providers available', provider: 'none', model: 'none', prompt, response: null };
        }

        const modelConfig = this.resolveModelConfig(resolvedProvider);

        try {
            const chain = this._getChain(resolvedProvider, sessionId, temperature, maxTokens);
            const result = await chain.invoke({ input: prompt }, { configurable: { sessionId } });
            const responseText = this._extractText(resolvedProvider, result);
            const responseMetadata = result?.response_metadata ?? result?.responseMetadata ?? null;
            const usageMetadata = result?.usage_metadata ?? result?.usageMetadata ?? null;
            const messageId = result?.id ?? null;
            const finishReason = responseMetadata?.finish_reason ?? responseMetadata?.stop_reason ?? responseMetadata?.finishReason ?? null;
            const tokenUsage = this._extractTokenUsage(responseMetadata, usageMetadata);

            return {
                success: true,
                provider: modelConfig.provider,
                model: modelConfig.model,
                modelIdentifier: modelConfig.name,
                prompt,
                response: responseText,
                temperature,
                maxTokens,
                sessionId,
                ...(responseMetadata != null ? { response_metadata: responseMetadata } : {}),
                ...(usageMetadata != null ? { usage_metadata: usageMetadata } : {}),
                ...(tokenUsage != null ? { token_usage: tokenUsage } : {}),
                ...(messageId != null ? { id: messageId } : {}),
                ...(finishReason != null ? { finish_reason: finishReason } : {}),
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
                sessionId,
            };
        }
    }

    async getHistory(provider, sessionId = 'default') {
        const history = this._getHistory(sessionId);
        const messages = await history.getMessages();
        const turns = messages.map((message) => {
            const additionalKwargs = message?.additional_kwargs ?? message?.additionalKwargs ?? {};
            const responseMetadata = message?.response_metadata ?? message?.responseMetadata ?? additionalKwargs?.response_metadata ?? null;
            const usageMetadata = message?.usage_metadata ?? message?.usageMetadata ?? additionalKwargs?.usage_metadata ?? null;
            const tokenUsage = this._extractTokenUsage(responseMetadata, usageMetadata);
            const turn = {
                role: message._getType?.() ?? message.getType?.() ?? message.type,
                content: message.content,
            };
            if (tokenUsage != null) turn.token_usage = tokenUsage;
            return turn;
        });
        return { provider, sessionId, turns, count: turns.length };
    }

    async resetMemory(provider = null, sessionId = null) {
        const removedSessions = [];

        if (sessionId) {
            const history = this.histories.get(sessionId);
            if (history) await history.clear();
            this.histories.delete(sessionId);
            for (const key of Array.from(this.chains.keys())) {
                if (key.endsWith(`::${sessionId}`)) this.chains.delete(key);
            }
            removedSessions.push(sessionId);
        } else {
            await Promise.all(Array.from(this.histories.values()).map((history) => history.clearAllSessions()));
            this.chains.clear();
            this.histories.clear();
            removedSessions.push('ALL');
        }

        const sessionsDir = path.join(__dirname, 'sessions');
        if (removedSessions.length === 1 && removedSessions[0] === 'ALL') {
            if (fs.existsSync(sessionsDir)) {
                for (const file of fs.readdirSync(sessionsDir)) {
                    if (file.endsWith('.json')) fs.unlinkSync(path.join(sessionsDir, file));
                }
            }
        } else {
            for (const session of removedSessions) {
                const file = this._sessionFilePath(session);
                if (fs.existsSync(file)) fs.unlinkSync(file);
            }
        }

        return { status: 'cleared', removedSessions };
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
