/**
 * utils.js - Common utilities and configurations shared across all JavaScript frameworks.
 */

import 'dotenv/config';
import fs from 'fs';
import readline from 'readline';
import { config } from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

import { normalizeResponseText } from './stream.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
config({ path: join(__dirname, '.env') });

export const PROVIDERS = {
    anthropic: { apiKeyEnv: 'ANTHROPIC_API_KEY', defaultModel: 'claude-sonnet-4-5', displayName: 'Anthropic Claude' },
    openai: { apiKeyEnv: 'OPENAI_API_KEY', defaultModel: 'gpt-5.2', displayName: 'OpenAI GPT' },
    google: { apiKeyEnv: 'GOOGLE_API_KEY', defaultModel: 'gemini-3-flash-preview', displayName: 'Google Gemini' },
    xai: { apiKeyEnv: 'XAI_API_KEY', defaultModel: 'grok-4', displayName: 'xAI Grok' },
};

export function getApiKey(provider) {
    return (provider in PROVIDERS) ? process.env[PROVIDERS[provider].apiKeyEnv] : null;
}

export function getDefaultModel(provider) {
    return PROVIDERS[provider]?.defaultModel || '';
}

export function getDisplayName(provider) {
    return PROVIDERS[provider]?.displayName || provider.charAt(0).toUpperCase() + provider.slice(1);
}

export function getAllProviders() {
    return Object.keys(PROVIDERS);
}

let sharedRl = null;
function getSharedAsk() {
    if (!sharedRl) {
        sharedRl = readline.createInterface({ input: process.stdin, output: process.stdout });
    }
    return (prompt) => new Promise((resolve) => sharedRl.question(prompt, resolve));
}



export function parseStructuredJsonResponse(raw) {
    let content = '';
    if (raw == null) content = '';
    else if (typeof raw === 'string') content = raw.trim();
    else if (typeof raw === 'object' && typeof raw.content === 'string') content = raw.content.trim();
    else if (typeof raw === 'object') content = JSON.stringify(raw);
    else content = String(raw).trim();

    const parts = content.match(/content=(["'])((?:\\.|(?!\1).)*)\1\s+additional_kwargs=/s);
    if (parts) {
        const quote = parts[1];
        content = parts[2];
        content = quote === "'" ? content.replace(/\\'/g, "'") : content.replace(/\\"/g, '"');
        content = content.trim();
    }

    if (content.startsWith('```json')) content = content.slice(7);
    if (content.startsWith('```')) content = content.slice(3);
    if (content.endsWith('```')) content = content.slice(0, -3);
    content = content.trim();

    try {
        const parsed = JSON.parse(content);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
    } catch {}

    // Some SDK responses are list-based content blocks serialized as repr-like text.
    if (content.startsWith('[') && content.endsWith(']')) {
        try {
            const blocks = Function(`"use strict"; return (${content});`)();
            if (Array.isArray(blocks)) {
                for (const block of blocks) {
                    if (block && typeof block === 'object') {
                        const maybeText = block.text || block.content;
                        if (typeof maybeText === 'string' && maybeText.trim()) {
                            content = maybeText.trim();
                            if (content.startsWith('```json')) content = content.slice(7);
                            if (content.startsWith('```')) content = content.slice(3);
                            if (content.endsWith('```')) content = content.slice(0, -3);
                            content = content.trim();
                            break;
                        }
                    }
                }
            }
        } catch {}
    }

    // Fallback: parse first complete JSON object from mixed text.
    const start = content.indexOf('{');
    if (start === -1) {
        const parsed = JSON.parse(content);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
        throw new Error('Parsed structured content is not a JSON object');
    }

    let depth = 0;
    let inString = false;
    let escape = false;

    for (let i = start; i < content.length; i += 1) {
        const ch = content[i];
        if (inString) {
            if (escape) escape = false;
            else if (ch === '\\') escape = true;
            else if (ch === '"') inString = false;
            continue;
        }

        if (ch === '"') {
            inString = true;
            continue;
        }

        if (ch === '{') depth += 1;
        else if (ch === '}') {
            depth -= 1;
            if (depth === 0) {
                const parsed = JSON.parse(content.slice(start, i + 1));
                if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
                throw new Error('Parsed structured content is not a JSON object');
            }
        }
    }

    const parsed = JSON.parse(content);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
    throw new Error('Parsed structured content is not a JSON object');
}

export async function getUserParameters(ask) {
    const tempInput = await ask('Temperature (0.0-2.0, default 0.7): ');
    let temperature = 0.7;
    if (tempInput.trim()) {
        const parsed = parseFloat(tempInput);
        temperature = !Number.isNaN(parsed) ? Math.max(0.0, Math.min(2.0, parsed)) : 0.7;
    }

    const tokensInput = await ask('Max tokens (default 1000): ');
    let maxTokens = 1000;
    if (tokensInput.trim()) {
        const parsed = parseInt(tokensInput, 10);
        maxTokens = !Number.isNaN(parsed) ? Math.max(1, Math.min(4000, parsed)) : 1000;
    }

    return { temperature, maxTokens };
}

export function displayProviderResponse(provider, response, framework = '') {
    const providerDisplay = `${getDisplayName(provider)}${framework ? ` (${framework})` : ''} answered:`;
    console.log(`\n=== ${providerDisplay} ===`);

    const configParts = [];
    if (typeof response.temperature !== 'undefined') configParts.push(`temp: ${response.temperature}`);
    if (typeof response.maxTokens !== 'undefined') configParts.push(`max_tokens: ${response.maxTokens}`);
    else if (typeof response.max_tokens !== 'undefined') configParts.push(`max_tokens: ${response.max_tokens}`);
    if (response.model) configParts.push(`model: ${response.model}`);
    if (configParts.length > 0) {
        console.log(`[${configParts.join(', ')}]`);
    }

    if (response.success) {
        const raw = response.response;
        if (raw && typeof raw === 'object') {
            console.log(JSON.stringify(raw, null, 2));
        } else {
            console.log(normalizeResponseText(raw) || 'No response');
        }
    } else {
        console.log(`Error: ${response.error || 'Unknown error'}`);
    }
    console.log('='.repeat(60));
}

export async function getUserChoice(options, prompt, ask) {
    console.log(`\n${prompt}`);
    options.forEach((option, i) => console.log(`${i + 1}. ${option}`));
    while (true) {
        const answer = (await ask(`Select an option (1-${options.length}, default 1): `)).trim();
        const choice = (answer === '' ? 1 : parseInt(answer, 10)) - 1;
        if (choice >= 0 && choice < options.length) return choice;
        console.log('Invalid selection. Please try again.');
    }
}

export function formatFilename(question, framework) {
    const safeQuestion = question.slice(0, 20).replace(/\s+/g, '_').replace(/[?!]/g, '');
    return `llm_responses_${framework}_${safeQuestion}.json`;
}

export function saveResponseToFile(response, filename) {
    fs.writeFileSync(filename, JSON.stringify(response, null, 2));
    console.log(`Response saved to ${filename}`);
}

export class BaseLLMManager {
    constructor(frameworkName) {
        this.framework = frameworkName;
        this.initializationMessages = {};
    }

    async _checkProviders() {
        for (const provider of getAllProviders()) {
            if (getApiKey(provider)) {
                try {
                    await this._testProvider(provider);
                    this.initializationMessages[provider] = '✓ Initialized successfully';
                } catch (error) {
                    this.initializationMessages[provider] = `✗ Failed: ${error.message}`;
                }
            } else {
                this.initializationMessages[provider] = '✗ API key not found';
            }
        }
    }

    getAvailableProviders() {
        return Object.entries(this.initializationMessages)
            .filter(([_, status]) => status.startsWith('✓'))
            .map(([provider]) => provider);
    }

    displayInitializationStatus() {
        console.log(`\n=== ${this.framework} Framework - Provider Status ===`);
        for (const [provider, message] of Object.entries(this.initializationMessages)) {
            console.log(`${getDisplayName(provider)}: ${message}`);
        }
        console.log('='.repeat(50) + '\n');
    }

    async queryAllProviders(topic, template = '{topic}', maxTokens = 1000, temperature = 0.7) {
        const available = this.getAvailableProviders();
        if (available.length === 0) {
            return { success: false, error: 'No providers available', prompt: template.replace('{topic}', topic), responses: {} };
        }

        const responses = {};
        for (const provider of available) {
            console.log(`Querying ${getDisplayName(provider)}...`);
            responses[provider] = await this.askQuestion(topic, provider, template, maxTokens, temperature);
        }

        return {
            success: true,
            prompt: template.replace('{topic}', topic),
            responses,
        };
    }
}

export async function interactiveCli(manager) {
    const ask = getSharedAsk();

    try {
        console.log('='.repeat(60));
        console.log(`LLM Application - ${manager.framework} Framework`);
        console.log('='.repeat(60));

        manager.displayInitializationStatus();
        const availableProviders = manager.getAvailableProviders();

        if (availableProviders.length === 0) {
            console.log('No providers available. Check your .env file.');
            return;
        }

        const { temperature, maxTokens } = await getUserParameters(ask);
        console.log(`\nUsing temperature: ${temperature}, max tokens: ${maxTokens}`);

        const sortedProviders = [...availableProviders].sort((a, b) => {
            if (a === 'openai') return -1;
            if (b === 'openai') return 1;
            return getDisplayName(a).localeCompare(getDisplayName(b));
        });
        console.log(`\nAvailable providers: ${sortedProviders.map((p) => getDisplayName(p)).join(', ')}`);
        const mode = ((await ask('Query ALL providers or select one? (all/one, default one): ')).trim().toLowerCase() || 'one');

        const fullMemorySupported = Boolean(
            manager.memoryEnabled
            && typeof manager.askQuestion === 'function'
            && typeof manager.getHistory === 'function'
            && typeof manager.resetMemory === 'function'
        );
        const retrievalMemorySupported = Boolean(
            manager.retrievalMemoryEnabled
            && typeof manager.askQuestion === 'function'
            && typeof manager.getHistory === 'function'
            && typeof manager.resetMemory === 'function'
        );
        const memorySupported = fullMemorySupported || retrievalMemorySupported;

        if (['all', 'a', ''].includes(mode)) {
            const question = await ask('Enter your question: ');
            const results = await manager.queryAllProviders(question, '{topic}', maxTokens, temperature);
            if (results.success) {
                for (const [provider, res] of Object.entries(results.responses)) {
                    displayProviderResponse(provider, res, manager.framework);
                }
            } else {
                console.log(`Error: ${results.error}`);
            }

            const save = (await ask('\nSave results? (y/n): ')).toLowerCase();
            if (save === 'y' || save === 'yes') {
                saveResponseToFile(results, formatFilename(question, manager.framework.toLowerCase()));
            }
        } else {
            const names = sortedProviders.map((p) => getDisplayName(p));
            const choice = await getUserChoice(names, 'Select a provider:', ask);
            const provider = sortedProviders[choice];

            let sessionId = 'default';
            if (memorySupported) {
                const sessionInput = (await ask("Enter memory session ID (default: 'default'): ")).trim();
                if (sessionInput) sessionId = sessionInput;
                console.log(`Using memory session: ${sessionId}`);
            }

            console.log('\n' + '='.repeat(50));
            console.log(`${manager.framework.toUpperCase()} INTERACTIVE MODE - ${getDisplayName(provider).toUpperCase()}`);
            console.log('='.repeat(50));

            while (true) {
                const prompt = memorySupported
                    ? "\nAsk a question (or 'history', 'clear', 'exit'): "
                    : "\nAsk a question (or 'exit'): ";
                const userInput = (await ask(prompt)).trim();

                if (['exit', 'quit'].includes(userInput.toLowerCase())) break;
                if (!userInput) continue;

                if (memorySupported && userInput.toLowerCase() === 'history') {
                    const history = await Promise.resolve(manager.getHistory(provider, sessionId));
                    console.log(`\n🧠 Memory for ${getDisplayName(provider)} (session: ${sessionId}):`);
                    for (const turn of history.turns) {
                        const role = (turn.role || 'unknown').toString();
                        console.log(`[${role.charAt(0).toUpperCase()}${role.slice(1)}] ${turn.content}`);
                    }
                    if (!history.turns.length) console.log('No memory yet.');
                    continue;
                }

                if (memorySupported && userInput.toLowerCase() === 'clear') {
                    await Promise.resolve(manager.resetMemory(provider, sessionId));
                    console.log(`✅ Memory cleared for session '${sessionId}'`);
                    continue;
                }

                const result = memorySupported
                    ? await manager.askQuestion(userInput, provider, '{topic}', maxTokens, temperature, sessionId)
                    : await manager.askQuestion(userInput, provider, '{topic}', maxTokens, temperature);

                displayProviderResponse(provider, result, manager.framework);
            }
        }
    } catch (error) {
        console.error('Fatal Error:', error);
    } finally {
        console.log(`\nThank you for using the ${manager.framework} LLM Application!`);
        if (sharedRl) {
            sharedRl.close();
            sharedRl = null;
        }
    }
}
