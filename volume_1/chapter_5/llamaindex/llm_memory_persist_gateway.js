/**
 * LLM Memory Gateway - LlamaIndex JS with persistent session memory.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { ChatMemoryBuffer } from '../../chapter_4/node_modules/@llamaindex/core/memory/dist/index.js';
import { LlamaIndexLLMManager as Chapter4LlamaIndexManager } from '../../chapter_4/llamaindex/llm_gateway.js';
import { getDefaultModel, interactiveCli } from '../../chapter_4/utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class LlamaIndexLLMManager extends Chapter4LlamaIndexManager {
    constructor(memoryEnabled = true) {
        super();
        this.framework = 'LlamaIndex+History JS';
        this.memoryEnabled = memoryEnabled;
        this.sessions = new Map();
    }

    _historyKey(provider, sessionId) {
        return `${provider}::${sessionId}`;
    }

    _sessionFilePath(provider, sessionId) {
        const sessionsDir = path.join(__dirname, 'sessions');
        fs.mkdirSync(sessionsDir, { recursive: true });
        return path.join(sessionsDir, `${provider}__${sessionId}.json`);
    }

    _normalizeTurns(rawTurns) {
        if (!Array.isArray(rawTurns)) return [];
        return rawTurns
            .map((turn) => ({
                role: turn?.role ?? turn?.message?.role,
                content: turn?.content ?? turn?.message?.content,
            }))
            .filter((turn) => typeof turn.role === 'string' && typeof turn.content === 'string');
    }

    _readSessionTurns(provider, sessionId) {
        const key = this._historyKey(provider, sessionId);
        if (this.sessions.has(key)) return this.sessions.get(key);

        const filePath = this._sessionFilePath(provider, sessionId);
        let turns = [];

        if (fs.existsSync(filePath)) {
            try {
                const payload = JSON.parse(fs.readFileSync(filePath, 'utf8'));
                if (Array.isArray(payload)) {
                    turns = this._normalizeTurns(payload);
                } else if (Array.isArray(payload?.turns)) {
                    turns = this._normalizeTurns(payload.turns);
                }
            } catch {
                turns = [];
            }
        }

        this.sessions.set(key, turns);
        return turns;
    }

    _persistSessionTurns(provider, sessionId) {
        const key = this._historyKey(provider, sessionId);
        const turns = this.sessions.get(key) ?? [];
        const filePath = this._sessionFilePath(provider, sessionId);
        fs.writeFileSync(filePath, JSON.stringify({ turns }, null, 2), 'utf8');
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
            const turns = this._readSessionTurns(resolvedProvider, sessionId);
            const messages = [...turns, { role: 'user', content: prompt }];
            const client = this._createClient(resolvedProvider, temperature, maxTokens);

            const result = await client.chat({ messages });
            const responseText = this._extractText(result);

            turns.push({ role: 'user', content: prompt });
            turns.push({ role: 'assistant', content: responseText });
            this._persistSessionTurns(resolvedProvider, sessionId);

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
        const turns = [...this._readSessionTurns(provider, sessionId)];
        return { provider, sessionId, turns, count: turns.length };
    }

    resetMemory(provider = null, sessionId = null) {
        const removedSessions = [];
        const sessionsDir = path.join(__dirname, 'sessions');

        if (provider && sessionId) {
            const key = this._historyKey(provider, sessionId);
            this.sessions.delete(key);
            removedSessions.push([provider, sessionId]);
        } else if (provider) {
            for (const key of Array.from(this.sessions.keys())) {
                const [p, s] = key.split('::');
                if (p === provider) {
                    this.sessions.delete(key);
                    removedSessions.push([p, s]);
                }
            }
        } else if (sessionId) {
            for (const key of Array.from(this.sessions.keys())) {
                const [p, s] = key.split('::');
                if (s === sessionId) {
                    this.sessions.delete(key);
                    removedSessions.push([p, s]);
                }
            }
        } else {
            this.sessions.clear();
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
