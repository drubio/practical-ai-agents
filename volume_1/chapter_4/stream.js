/** Streaming helpers for chapter web APIs. */

function extractPythonStyleContent(payload) {
    const marker = 'content=';
    const start = payload.indexOf(marker);
    if (start < 0) return null;

    const quoteIndex = start + marker.length;
    const quote = payload[quoteIndex];
    if (quote !== '"' && quote !== "'") return null;

    let i = quoteIndex + 1;
    let escaped = false;
    let value = '';

    while (i < payload.length) {
        const char = payload[i];
        if (escaped) {
            switch (char) {
                case 'n':
                    value += '\n';
                    break;
                case 'r':
                    value += '\r';
                    break;
                case 't':
                    value += '\t';
                    break;
                default:
                    value += char;
                    break;
            }
            escaped = false;
            i += 1;
            continue;
        }

        if (char === '\\') {
            escaped = true;
            i += 1;
            continue;
        }

        if (char === quote) {
            const remaining = payload.slice(i + 1);
            if (remaining.includes('additional_kwargs=')) {
                return value;
            }
            return null;
        }

        value += char;
        i += 1;
    }

    return null;
}

export function normalizeResponseText(payload) {
    if (payload == null) return '';

    if (typeof payload === 'string') {
        const extracted = extractPythonStyleContent(payload);
        if (typeof extracted === 'string') {
            return extracted;
        }

        try {
            const maybeJson = JSON.parse(payload);
            if (maybeJson && typeof maybeJson === 'object') {
                for (const key of ['answer', 'distilled', 'content', 'text', 'message', 'summary']) {
                    const value = maybeJson[key];
                    if (typeof value === 'string' && value.trim()) return value;
                }
            }
        } catch {
            // not JSON text, keep original payload
        }

        return payload;
    }

    if (typeof payload === 'object') {
        for (const key of ['content', 'text', 'message', 'answer', 'final_answer', 'distilled', 'summary']) {
            const value = payload[key];
            if (typeof value === 'string' && value.trim()) return value;
        }
        return JSON.stringify(payload);
    }

    return String(payload);
}

export function chunkText(text, chunkSize = 28) {
    const clean = text || '';
    if (!clean) return [''];

    const chunks = [];
    for (let index = 0; index < clean.length; index += chunkSize) {
        chunks.push(clean.slice(index, index + chunkSize));
    }
    return chunks;
}
