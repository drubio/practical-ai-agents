/**
 * web.js - Clean web interface for JavaScript LLM testers.
 * Supports optional memory endpoints and session-based tracking.
 */

import express from 'express';
import cors from 'cors';
import { getDisplayName, getDefaultModel } from './utils.js';
import { chunkText, normalizeResponseText } from './stream.js';

async function captureConsoleOutputAsync(fn) {
    const originalLog = console.log;
    const logs = [];
    console.log = (...args) => logs.push(args.join(' '));
    try {
        const result = await fn();
        return { result, logs: logs.join('\n') };
    } finally {
        console.log = originalLog;
    }
}

function supportsMemory(manager) {
    return Boolean(
        manager?.memoryEnabled
        && typeof manager.getHistory === 'function'
        && typeof manager.resetMemory === 'function'
    );
}



function parseStructuredRawResponse(rawResponse) {
    if (rawResponse && typeof rawResponse === 'object' && !Array.isArray(rawResponse)) {
        return rawResponse;
    }
    if (typeof rawResponse !== 'string') {
        return null;
    }

    const trimmed = rawResponse.trim();
    if (!trimmed) {
        return null;
    }

    try {
        const parsed = JSON.parse(trimmed);
        return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null;
    } catch {
        if (trimmed.startsWith('```json') && trimmed.endsWith('```')) {
            try {
                const parsed = JSON.parse(trimmed.slice(7, -3).trim());
                return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null;
            } catch {
                return null;
            }
        }
        return null;
    }
}

function extractAnswerText(responseObj, rawResponse) {
    if (responseObj && typeof responseObj === 'object') {
        const answer = responseObj.answer || responseObj.distilled || responseObj.summary;
        if (typeof answer === 'string' && answer.trim().length > 0) {
            return answer;
        }
    }
    return normalizeResponseText(rawResponse);
}

function recoverStructuredParseError(result) {
    if (!result) {
        return result;
    }

    const rawResponse = result.rawResponse;
    const parsedResponse = parseStructuredRawResponse(rawResponse);

    if (result.success) {
        if (parsedResponse && result.response && typeof result.response === 'object') {
            const metadataNotes = result.response?.metadata?.notes;
            if (typeof metadataNotes === 'string' && metadataNotes.startsWith('Failed to parse structured JSON response')) {
                return {
                    ...result,
                    response: parsedResponse,
                    rawAnswer: extractAnswerText(parsedResponse, rawResponse),
                };
            }
        }
        return result;
    }

    const errorMessage = String(result.error || '');
    if (!errorMessage.includes('Failed to parse structured JSON response') || !rawResponse) {
        return result;
    }

    const recoveredResponse = parsedResponse || {
        answer: normalizeResponseText(rawResponse),
        distilled: normalizeResponseText(rawResponse),
        metadata: {
            confidence: 'low',
            notes: errorMessage,
        },
    };

    return {
        ...result,
        success: true,
        error: null,
        response: recoveredResponse,
        rawAnswer: extractAnswerText(recoveredResponse, rawResponse),
    };
}

function buildManager(managerClassOrFactory) {
    if (typeof managerClassOrFactory !== 'function') {
        throw new Error('Invalid manager class/factory provided');
    }
    try {
        return new managerClassOrFactory();
    } catch {
        return managerClassOrFactory();
    }
}

function createWebApi(managerClassOrFactory) {
    const app = express();
    app.use(cors());
    app.use(express.json());

    let manager;
    const initPromise = (async () => {
        manager = buildManager(managerClassOrFactory);
        await manager._checkProviders();
        return manager;
    })();

    app.use(async (_, __, next) => {
        if (!manager) {
            await initPromise;
        }
        next();
    });

    app.get('/', async (_, res) => {
        const available = manager.getAvailableProviders();
        res.json({
            framework: manager.framework,
            available_providers: available,
            total_available: available.length,
            initialization_status: manager.initializationMessages,
            status: available.length > 0 ? 'healthy' : 'no_providers',
        });
    });

    app.get('/providers', async (_, res) => {
        const available = manager.getAvailableProviders();
        res.json({
            framework: manager.framework,
            providers: available.map((provider) => ({
                name: provider,
                display_name: getDisplayName(provider),
                model: getDefaultModel(provider),
                status: manager.initializationMessages[provider] || 'Unknown',
            })),
            count: available.length,
        });
    });

    app.get('/capabilities', async (_, res) => {
        res.json({
            framework: manager.framework,
            streaming: true,
            memory: supportsMemory(manager),
        });
    });

    app.post('/query', async (req, res) => {
        try {
            const {
                topic,
                provider = null,
                template = '{topic}',
                max_tokens = 1000,
                temperature = 0.7,
                session_id = 'default',
                sessionId = null,
            } = req.body;

            if (!topic) {
                return res.status(400).json({ error: 'Topic is required' });
            }

            const effectiveSessionId = sessionId ?? session_id ?? 'default';
            const { result, logs } = await captureConsoleOutputAsync(async () => {
                if (supportsMemory(manager)) {
                    return recoverStructuredParseError(
                        await manager.askQuestion(topic, provider, template, max_tokens, temperature, effectiveSessionId)
                    );
                }
                return recoverStructuredParseError(
                    await manager.askQuestion(topic, provider, template, max_tokens, temperature)
                );
            });

            if (!result.success) {
                return res.status(400).json({
                    error: result.error || 'Query failed',
                    provider: result.provider,
                    debug: logs || null,
                });
            }

            return res.json({
                success: true,
                framework: manager.framework,
                provider: result.provider,
                model: result.model,
                response: (typeof result.response === 'object' && result.response !== null)
                    ? result.response
                    : normalizeResponseText(result.response),
                parameters: {
                    temperature: result.temperature,
                    max_tokens: result.maxTokens,
                    template,
                },
                prompt: result.prompt,
                session_id: result.sessionId || effectiveSessionId,
                ...(logs ? { debug: logs } : {}),
            });
        } catch (error) {
            return res.status(500).json({ error: error.message, framework: manager.framework });
        }
    });

    app.post('/query-stream', async (req, res) => {
        try {
            const {
                topic,
                provider = null,
                template = '{topic}',
                max_tokens = 1000,
                temperature = 0.7,
                session_id = 'default',
                sessionId = null,
            } = req.body;

            if (!topic) {
                return res.status(400).json({ error: 'Topic is required' });
            }

            const effectiveSessionId = sessionId ?? session_id ?? 'default';
            const { result } = await captureConsoleOutputAsync(async () => {
                if (supportsMemory(manager)) {
                    return recoverStructuredParseError(
                        await manager.askQuestion(topic, provider, template, max_tokens, temperature, effectiveSessionId)
                    );
                }
                return recoverStructuredParseError(
                    await manager.askQuestion(topic, provider, template, max_tokens, temperature)
                );
            });

            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');

            if (!result.success) {
                res.write(`data: ${JSON.stringify({ type: 'error', error: result.error || 'Query failed' })}\n\n`);
                return res.end();
            }

            const responseText = normalizeResponseText(result.response);
            for (const chunk of chunkText(responseText)) {
                res.write(`data: ${JSON.stringify({ type: 'chunk', content: chunk })}\n\n`);
                await new Promise((resolve) => setTimeout(resolve, 35));
            }

            res.write(`data: ${JSON.stringify({ type: 'done', provider: result.provider, model: result.model, response: (typeof result.response === 'object' && result.response !== null) ? result.response : null, token_usage: result.tokenUsage ?? result.token_usage ?? null, session_id: result.sessionId ?? effectiveSessionId })}\n\n`);
            return res.end();
        } catch (error) {
            if (!res.headersSent) {
                return res.status(500).json({ error: error.message, framework: manager.framework });
            }
            res.write(`data: ${JSON.stringify({ type: 'error', error: error.message })}\n\n`);
            return res.end();
        }
    });

    app.post('/query-all', async (req, res) => {
        try {
            const { topic, template = '{topic}', max_tokens = 1000, temperature = 0.7 } = req.body;
            if (!topic) {
                return res.status(400).json({ error: 'Topic is required' });
            }

            const { result, logs } = await captureConsoleOutputAsync(async () => (
                manager.queryAllProviders(topic, template, max_tokens, temperature)
            ));

            if (!result.success) {
                return res.status(400).json({
                    error: result.error || 'Query failed',
                    framework: manager.framework,
                    debug: logs || null,
                });
            }

            const cleanResponses = {};
            let successful = 0;
            let failed = 0;
            for (const [provider, response] of Object.entries(result.responses || {})) {
                if (response.success) {
                    cleanResponses[provider] = {
                        success: true,
                        response: (typeof response.response === 'object' && response.response !== null)
                            ? response.response
                            : normalizeResponseText(response.response),
                        model: response.model,
                        parameters: {
                            temperature: response.temperature,
                            max_tokens: response.maxTokens,
                        },
                    };
                    successful += 1;
                } else {
                    cleanResponses[provider] = {
                        success: false,
                        error: response.error || 'Unknown error',
                        model: response.model || 'unknown',
                    };
                    failed += 1;
                }
            }

            return res.json({
                success: true,
                framework: manager.framework,
                prompt: result.prompt,
                responses: cleanResponses,
                summary: {
                    total_providers: Object.keys(result.responses || {}).length,
                    successful,
                    failed,
                },
                parameters: { temperature, max_tokens, template },
                ...(logs ? { debug: logs } : {}),
            });
        } catch (error) {
            return res.status(500).json({ error: error.message, framework: manager.framework });
        }
    });

    app.get('/history', async (req, res) => {
        if (!supportsMemory(manager)) {
            return res.status(400).json({ error: 'Memory not supported by this manager' });
        }
        const { provider = 'openai', session_id = 'default', sessionId = null } = req.query;
        return res.json(manager.getHistory(provider, sessionId ?? session_id ?? "default"));
    });

    app.post('/reset-memory', async (req, res) => {
        if (!supportsMemory(manager)) {
            return res.status(400).json({ error: 'Memory not supported by this manager' });
        }
        const body = req.body || {};
        const bodyProvider = body.provider ?? null;
        const bodySessionId = body.sessionId ?? body.session_id ?? null;
        const queryProvider = req.query?.provider ?? null;
        const querySessionId = req.query?.sessionId ?? req.query?.session_id ?? null;
        const provider = bodyProvider !== null ? bodyProvider : queryProvider;
        const sessionId = bodySessionId !== null ? bodySessionId : querySessionId;
        return res.json(manager.resetMemory(provider, sessionId));
    });

    app.get('/health', async (_, res) => {
        const available = manager.getAvailableProviders();
        res.json({
            status: available.length > 0 ? 'healthy' : 'unhealthy',
            framework: manager.framework,
            providers_available: available.length,
        });
    });

    return app;
}

export async function runWebServer(managerClassOrFactory, host = '0.0.0.0', port = 8000) {
    const app = createWebApi(managerClassOrFactory);

    let frameworkName = 'Unknown';
    try {
        frameworkName = buildManager(managerClassOrFactory).framework;
    } catch {
        // Ignore display failure
    }

    app.listen(port, host, () => {
        console.log(`Starting web server for ${frameworkName} framework...`);
        console.log(`Health check: http://${host}:${port}/health`);
        console.log(`Status: http://${host}:${port}/`);
    });
}

function main() {
    console.log('Universal LLM Web API (JavaScript)');
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
