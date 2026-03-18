'use client';

import { useMemo, useRef, useState } from 'react';
import { Client as LangGraphClient } from '@langchain/langgraph-sdk';
import {
  ChatMessage,
  FrameworkHeader,
  SettingsSidebar,
  ThinkingIndicator,
  ResponseDetailsPanel,
  CoagentActivitySidebar,
  CoagentCallDetails,
  useAPISettings,
  callAPI,
  getErrorMessage,
  formatHistoryMessage,
  getHistory,
  resetMemory,
  createMessageId,
  LANGGRAPH_API_URL,
  LANGGRAPH_ASSISTANT_ID
} from './shared';

type StreamEvent = {
  data?: unknown;
};

type LangGraphClientShape = {
  threads?: {
    create?: () => Promise<{ thread_id?: string; threadId?: string }>;
  };
  runs?: {
    stream?: (
      threadId: string,
      assistantId: string,
      payload: {
        input: { messages: Array<{ role: string; content: string }> };
        streamMode: string;
        config: {
          configurable: { provider: string; session_id: string };
          temperature: number;
          max_tokens: number;
        };
      }
    ) => Promise<AsyncIterable<StreamEvent>>;
  };
};

const getChunkText = (event: StreamEvent): string => {
  const payload = event?.data;

  if (typeof payload === 'string') return payload;

  if (Array.isArray(payload)) {
    return payload
      .map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object' && 'text' in item && typeof (item as { text?: unknown }).text === 'string') {
          return (item as { text: string }).text;
        }
        return '';
      })
      .filter(Boolean)
      .join('');
  }

  if (payload && typeof payload === 'object' && 'messages' in payload) {
    const messages = (payload as { messages?: Array<{ role?: string; content?: unknown }> }).messages ?? [];
    const assistantMessage = [...messages].reverse().find((message) => message.role === 'assistant');
    if (typeof assistantMessage?.content === 'string') return assistantMessage.content;
  }

  return '';
};

const isUnsupportedLangGraphEndpointError = (error: unknown) => {
  const message = getErrorMessage(error).toLowerCase();
  return message.includes('404') || message.includes('/threads') || message.includes('not found');
};

const LangGraphChatPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const { providers, settings, setSettings, apiStatus, checkApiStatus, apiCapabilities } = useAPISettings();
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 1, role: 'assistant', content: 'Hello! I\'m your LangGraph assistant connected to your application web based API.' }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeCoagentCalls, setActiveCoagentCalls] = useState<CoagentCallDetails[]>([]);
  const threadIdRef = useRef<string | null>(null);

  const langGraphSdkUrl = (process.env.NEXT_PUBLIC_LANGGRAPH_PLATFORM_API_URL || LANGGRAPH_API_URL).trim();
  const langGraphClient = useMemo(() => new LangGraphClient({ apiUrl: langGraphSdkUrl }), [langGraphSdkUrl]);

  const updateAssistantMessage = (assistantId: number, update: (message: ChatMessage) => ChatMessage) => {
    setMessages((prev) => prev.map((message) => (message.id === assistantId ? update(message) : message)));
  };

  const runWithLangGraphSdk = async (prompt: string, assistantId: number): Promise<boolean> => {
    if (settings.queryMode !== 'single' || settings.responseMode !== 'stream') {
      return false;
    }

    try {
      const client = langGraphClient as unknown as LangGraphClientShape;
      if (!client.threads?.create || !client.runs?.stream) {
        return false;
      }

      if (!threadIdRef.current) {
        const thread = await client.threads.create();
        threadIdRef.current = thread.thread_id ?? thread.threadId ?? null;
      }

      if (!threadIdRef.current) return false;

      const stream = await client.runs.stream(threadIdRef.current, LANGGRAPH_ASSISTANT_ID, {
        input: { messages: [{ role: 'user', content: prompt }] },
        streamMode: 'messages-tuple',
        config: {
          configurable: {
            provider: settings.selectedProvider,
            session_id: settings.sessionId
          },
          temperature: settings.temperature,
          max_tokens: settings.maxTokens
        }
      });

      let gotText = false;
      for await (const chunk of stream) {
        const text = getChunkText(chunk);
        if (!text) continue;

        gotText = true;
        updateAssistantMessage(assistantId, (message) => ({ ...message, content: text }));
      }

      return gotText;
    } catch (error) {
      if (isUnsupportedLangGraphEndpointError(error)) {
        return false;
      }

      throw error;
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const userInput = input.trim();
    if (!userInput || isLoading) return;

    const userMessage: ChatMessage = { id: Date.now(), role: 'user', content: userInput };
    const assistantId = Date.now() + 1;
    setMessages((prev) => [...prev, userMessage, { id: assistantId, role: 'assistant', content: '' }]);
    setIsLoading(true);
    setInput('');

    try {
      const usedSdk = await runWithLangGraphSdk(userInput, assistantId);

      if (!usedSdk) {
        const response = await callAPI(userInput, settings, {
          onChunk: (partialText) => {
            updateAssistantMessage(assistantId, (message) => ({ ...message, content: partialText }));
          }
        });

        updateAssistantMessage(assistantId, (message) => ({
          ...message,
          content: response.content,
          details: response.details
        }));
        setActiveCoagentCalls(response.details?.coagentCalls || []);
      }
    } catch (error) {
      updateAssistantMessage(assistantId, (message) => ({
        ...message,
        content: `Connection Error: ${getErrorMessage(error)}`
      }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleHistoryAction = async (action: 'show' | 'reset', provider: string, sessionId: string) => {
    if (action === 'show') {
      try {
        const historyData = await getHistory(provider, sessionId);
        const historyContent = formatHistoryMessage(provider, sessionId, historyData.turns || []);
        setMessages((prev) => [...prev, { id: createMessageId(), role: 'system', content: historyContent }]);
      } catch (error) {
        setMessages((prev) => [...prev, { id: createMessageId(), role: 'system', content: `Error getting history: ${getErrorMessage(error)}` }]);
      }
      return;
    }

    try {
      await resetMemory(provider, sessionId);
      setMessages((prev) => [...prev, { id: createMessageId(), role: 'system', content: `✅ Memory cleared for ${provider} (${sessionId})` }]);
    } catch (error) {
      setMessages((prev) => [...prev, { id: createMessageId(), role: 'system', content: `Error resetting memory: ${getErrorMessage(error)}` }]);
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <FrameworkHeader
        title="LangGraph UI (SDK-first + application adapter fallback)"
        color="blue"
        settings={settings}
        onSettingsClick={() => setShowSettings(!showSettings)}
        apiStatus={apiStatus}
        apiCapabilities={apiCapabilities}
      />

      <div className="flex-1 overflow-hidden p-4">
        <div className={`grid h-full gap-4 ${apiCapabilities.hasCoagent ? 'lg:grid-cols-[minmax(0,1fr)_24rem]' : ''}`}>
          <div className="h-full bg-white rounded-lg border overflow-hidden flex flex-col">
          <div className="flex-1 overflow-y-auto p-4">
            {messages?.map((message) => (
              <div key={message.id} className={`mb-4 ${message.role === 'user' ? 'text-right' : message.role === 'system' ? 'text-center' : 'text-left'}`}>
                <div className={`inline-block px-4 py-2 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white max-w-md'
                    : message.role === 'system'
                    ? 'bg-amber-50 text-amber-900 border border-amber-200 max-w-3xl text-left whitespace-pre-wrap'
                    : 'bg-gray-100 text-gray-900 max-w-md whitespace-pre-wrap'
                }`}>
                  {message.role === 'system' && (
                    <div className="text-xs font-semibold uppercase tracking-wide text-amber-700 mb-2">Conversation History</div>
                  )}
                  {message.role === 'assistant' && !message.content && isLoading ? (
                    <ThinkingIndicator label="LangGraph chat processing..." />
                  ) : (
                    message.content
                  )}
                  {message.role === 'assistant' && <ResponseDetailsPanel details={message.details} />}
                </div>
              </div>
            ))}
          </div>

          <div className="border-t p-4">
            <form onSubmit={handleSubmit}>
              <div className="flex space-x-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  type="text"
                  placeholder={`Ask questions... (${settings.queryMode === 'single' ? settings.selectedProvider : 'all providers'})`}
                  className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-md"
                >
                  Send
                </button>
              </div>
            </form>
            <div className="text-xs text-gray-500 mt-2">
              LangGraph SDK URL: {langGraphSdkUrl} • assistant: {LANGGRAPH_ASSISTANT_ID} • endpoint adapter: {LANGGRAPH_API_URL}
            </div>
          </div>
          </div>

          {apiCapabilities.hasCoagent && (
            <div className="min-h-[18rem] lg:min-h-0">
              <CoagentActivitySidebar coagentCalls={activeCoagentCalls} />
            </div>
          )}
        </div>
      </div>

      <SettingsSidebar
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        settings={settings}
        onSettingsChange={setSettings}
        providers={providers}
        apiStatus={apiStatus}
        onRefreshApi={checkApiStatus}
        apiCapabilities={apiCapabilities}
        onHistoryAction={handleHistoryAction}
      />

      {showSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-40" onClick={() => setShowSettings(false)} />
      )}
    </div>
  );
};

export default LangGraphChatPage;
