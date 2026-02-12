import { NextRequest } from 'next/server';

type IncomingMessage = {
  role?: 'system' | 'user' | 'assistant';
  content?: string;
  parts?: Array<{ type?: string; text?: string }>;
};

async function readSSEToText(response: Response): Promise<string> {
  if (!response.body) {
    throw new Error('No response body available for streaming');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fullText = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() || '';

    for (const event of events) {
      const lines = event.split('\n').filter((line) => line.startsWith('data: '));
      for (const line of lines) {
        const payload = JSON.parse(line.slice(6));
        if (payload.type === 'chunk') {
          fullText += payload.content || '';
        } else if (payload.type === 'error') {
          throw new Error(payload.error || 'Streaming failed');
        }
      }
    }
  }

  return fullText;
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
      const content = await readSSEToText(response);
      return new Response(content, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
    }

    const data = await response.json();
    const content = data.success ? (typeof data.response === 'string' ? data.response : JSON.stringify(data.response)) : `Error: ${data.error}`;
    return new Response(content, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
  } catch (error) {
    return new Response(`Error: ${(error as Error).message}`, { status: 500 });
  }
}
