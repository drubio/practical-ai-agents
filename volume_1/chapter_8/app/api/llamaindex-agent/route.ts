import { NextRequest } from 'next/server';
import {
  buildDetails,
  extractApiErrorMessage,
  extractDetailsFromContent,
  getDisplayContentFromStructuredText,
  mergeDetails,
  processApiResponse,
  readResponsePayload,
  type ProcessedResponse,
  type ResponseDetails,
} from '../../lib/response-normalizer';

type IncomingMessage = {
  role?: 'system' | 'user' | 'assistant';
  content?: string;
  parts?: Array<{ type?: string; text?: string }>;
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

  const extractedDetails = extractDetailsFromContent(fullText);
  return { content: fullText, details: mergeDetails(streamDetails, extractedDetails) };
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
