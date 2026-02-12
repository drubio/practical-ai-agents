import { NextRequest } from 'next/server';

type IncomingMessage = {
  role?: 'system' | 'user' | 'assistant';
  content?: string;
  parts?: Array<{ type?: string; text?: string }>;
};

type TokenUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

type ResponseDetails = {
  provider?: string;
  sessionId?: string;
  summary?: string;
  distilled?: string;
  keywords?: string[];
  confidence?: string;
  notes?: string;
  tokenUsage?: TokenUsage;
};

type ProcessedResponse = {
  content: string;
  details?: ResponseDetails;
};

const parseJsonText = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) return null;

  try {
    return JSON.parse(trimmed);
  } catch (_) {
    const fenced = trimmed.match(/```json\s*([\s\S]*?)```/i);
    if (!fenced?.[1]) return null;
    try {
      return JSON.parse(fenced[1].trim());
    } catch {
      return null;
    }
  }
};

const extractTokenUsage = (data: any): TokenUsage | undefined => {
  const usage = data?.token_usage
    || data?.response?.token_usage
    || data?.turns?.findLast?.((turn: any) => turn?.token_usage)?.token_usage;

  if (!usage || typeof usage !== 'object') return undefined;
  return {
    prompt_tokens: usage.prompt_tokens,
    completion_tokens: usage.completion_tokens,
    total_tokens: usage.total_tokens
  };
};

const buildDetails = (structured: any, data: any): ResponseDetails | undefined => {
  if (!structured && !data) return undefined;

  const details: ResponseDetails = {
    provider: data?.provider,
    sessionId: data?.session_id,
    summary: structured?.summary,
    distilled: structured?.distilled,
    keywords: Array.isArray(structured?.keywords) ? structured.keywords.filter(Boolean) : undefined,
    confidence: structured?.metadata?.confidence,
    notes: structured?.metadata?.notes,
    tokenUsage: extractTokenUsage(data)
  };

  return Object.values(details).some((value) => value !== undefined) ? details : undefined;
};

const extractApiErrorMessage = (data: any, status?: number): string => {
  if (typeof data === 'string' && data.trim()) {
    return data.trim();
  }

  if (!data || typeof data !== 'object') {
    return status ? `Request failed with status ${status}` : 'Request failed';
  }

  const candidates = [
    data.error,
    data.message,
    data.detail,
    data.response?.error,
    data.response?.message,
    data.response?.detail
  ];

  const found = candidates.find((value) => typeof value === 'string' && value.trim().length > 0);
  if (found) return found;

  if (Array.isArray(data.detail)) {
    const details = data.detail
      .map((item: any) => {
        if (typeof item === 'string') return item;
        if (item?.msg) return String(item.msg);
        return '';
      })
      .filter(Boolean)
      .join('; ');

    if (details) return details;
  }

  return status ? `Request failed with status ${status}` : 'Request failed';
};

const readResponsePayload = async (response: Response): Promise<{ data: any; isJson: boolean }> => {
  const rawText = await response.text();

  if (!rawText.trim()) {
    return { data: null, isJson: false };
  }

  try {
    return { data: JSON.parse(rawText), isJson: true };
  } catch {
    return { data: rawText, isJson: false };
  }
};

const processApiResponse = (data: any, queryMode: string): ProcessedResponse => {
  if (queryMode === 'single') {
    if (data.success) {
      if (typeof data.response === 'string') {
        const structured = parseJsonText(data.response);
        return {
          content: structured?.answer || data.response,
          details: buildDetails(structured, data)
        };
      }

      if (typeof data.response === 'object' && data.response !== null) {
        return {
          content: data.response.answer || data.raw_answer || 'No response content available',
          details: buildDetails(data.response, data)
        };
      }

      return { content: data.raw_answer || 'No response available', details: buildDetails(undefined, data) };
    }

    return { content: `Error: ${extractApiErrorMessage(data)}` };
  }

  if (data.success) {
    let content = `Results from ${data.summary?.total_providers || Object.keys(data.responses).length} providers:\n\n`;

    for (const [provider, response] of Object.entries(data.responses)) {
      const providerResponse = response as any;
      content += `**${provider}**: `;

      if (providerResponse.success) {
        if (typeof providerResponse.response === 'string') {
          content += providerResponse.response;
        } else if (typeof providerResponse.response === 'object' && providerResponse.response !== null) {
          content += providerResponse.response.answer || providerResponse.raw_answer || 'No response available';
        } else {
          content += providerResponse.raw_answer || 'No response available';
        }
      } else {
        content += `Error: ${providerResponse.error}`;
      }

      content += '\n\n';
    }

    return { content };
  }

  return { content: `Error: ${extractApiErrorMessage(data)}` };
};

const getDisplayContentFromStructuredText = (content: string): string => {
  const parsed = parseJsonText(content);
  if (!parsed || typeof parsed !== 'object') return content;
  if (typeof parsed.answer === 'string' && parsed.answer.trim()) return parsed.answer;
  if (typeof parsed.distilled === 'string' && parsed.distilled.trim()) return parsed.distilled;
  return content;
};

async function readSSEToText(response: Response): Promise<{ content: string; details?: ResponseDetails }> {
  if (!response.body) {
    throw new Error('No response body available for streaming');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fullText = '';
  let streamDetails: ResponseDetails | undefined;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() || '';

    for (const event of events) {
      const lines = event.split('\n').filter((line) => line.startsWith('data: '));
      for (const line of lines) {
        let payload: any;
        try {
          payload = JSON.parse(line.slice(6));
        } catch {
          continue;
        }

        if (payload.type === 'chunk') {
          fullText += payload.content || '';
        } else if (payload.type === 'done') {
          streamDetails = buildDetails(payload.response, payload);
        } else if (payload.type === 'error') {
          throw new Error(payload.error || 'Streaming failed');
        }
      }
    }
  }

  return { content: fullText, details: streamDetails || buildDetails(parseJsonText(fullText), undefined) };
}

function getLastUserMessageText(messages: IncomingMessage[]): string {
  const lastUserMessage = [...messages].reverse().find((message) => message.role === 'user');

  if (!lastUserMessage) {
    return '';
  }

  if (typeof lastUserMessage.content === 'string' && lastUserMessage.content.trim().length > 0) {
    return lastUserMessage.content;
  }

  if (Array.isArray(lastUserMessage.parts)) {
    return lastUserMessage.parts
      .filter((part) => part?.type === 'text')
      .map((part) => part.text ?? '')
      .join('')
      .trim();
  }

  return '';
}

export async function POST(req: NextRequest) {
  try {
    const { messages = [], queryMode, selectedProvider, temperature, maxTokens, sessionId, responseMode } = await req.json();
    const topic = getLastUserMessageText(messages as IncomingMessage[]);

    if (!topic) {
      return new Response('Error: Missing user message content', { status: 400 });
    }

    const payload = {
      topic,
      temperature: temperature || 0.7,
      max_tokens: maxTokens || 1000,
      template: '{topic}',
      session_id: sessionId || 'default',
      ...(queryMode === 'single' && { provider: selectedProvider })
    };

    const useStreaming = queryMode === 'single' && responseMode === 'stream';
    const endpoint = useStreaming ? '/query-stream' : (queryMode === 'single' ? '/query' : '/query-all');

    const response = await fetch(`http://localhost:8000${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (useStreaming) {
      try {
        const streamResult = await readSSEToText(response);
        const result: ProcessedResponse = {
          content: getDisplayContentFromStructuredText(streamResult.content),
          details: streamResult.details
        };
        return new Response(JSON.stringify(result), { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
      } catch (streamError) {
        console.warn('Streaming proxy failed; falling back to standard query response.', streamError);

        const fallbackResponse = await fetch('http://localhost:8000/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const { data: fallbackData } = await readResponsePayload(fallbackResponse);
        if (!fallbackResponse.ok) {
          return new Response(
            JSON.stringify({ content: `Error: ${extractApiErrorMessage(fallbackData, fallbackResponse.status)}` }),
            { headers: { 'Content-Type': 'text/plain; charset=utf-8' } }
          );
        }

        const fallbackResult = processApiResponse(fallbackData, 'single');
        return new Response(JSON.stringify(fallbackResult), { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
      }
    }

    const { data } = await readResponsePayload(response);
    if (!response.ok) {
      return new Response(
        JSON.stringify({ content: `Error: ${extractApiErrorMessage(data, response.status)}` }),
        { headers: { 'Content-Type': 'text/plain; charset=utf-8' } }
      );
    }

    const result = processApiResponse(data, queryMode);
    return new Response(JSON.stringify(result), { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
  } catch (error) {
    return new Response(`Error: ${(error as Error).message}`, { status: 500 });
  }
}
