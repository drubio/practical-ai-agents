'use client';

import React, { useState, useMemo } from 'react';
import { Client as LangGraphClient } from '@langchain/langgraph-sdk';
import {
  APICapabilities,
  ChatMessage,
  ResponseDetails,
  FrameworkHeader,
  SettingsSidebar,
  ThinkingIndicator,
  ResponseDetailsPanel,
  useAPISettings,
  callAPI,
  getErrorMessage,
  formatHistoryMessage,
  getHistory,
  resetMemory,
  createMessageId,
  parseMessageEnvelope,
  LANGGRAPH_API_URL,
  LANGGRAPH_ASSISTANT_ID
} from './shared';

const LangGraphChatPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const { providers, settings, setSettings, apiStatus, checkApiStatus, apiCapabilities } = useAPISettings();
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 1, role: 'assistant', content: 'Hello! I\'m your LangGraph assistant connected to your gateway API.' }
  ]);
  const [isLoading, setIsLoading] = useState(false);

  // Keep LangGraph SDK in use and instantiate explicitly with URL params (no env vars required)
  const langGraphClient = useMemo(
    () => new LangGraphClient({ apiUrl: LANGGRAPH_API_URL }),
    []
  );

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input?.trim() || isLoading) return;

    const userMessage: ChatMessage = { id: Date.now(), role: 'user', content: input };
    const assistantId = Date.now() + 1;
    setMessages(prev => [...prev, userMessage, { id: assistantId, role: 'assistant', content: '' }]);
    setIsLoading(true);
    setInput('');

    try {
      const response = await callAPI(input, settings, {
        onChunk: (partialText) => {
          setMessages(prev => prev.map((message) => (
            message.id === assistantId ? { ...message, content: partialText } : message
          )));
        }
      });
      setMessages(prev => prev.map((message) => (
        message.id === assistantId ? { ...message, content: response.content, details: response.details } : message
      )));
    } catch (error) {
      setMessages(prev => prev.map((message) => (
        message.id === assistantId ? { ...message, content: `Connection Error: ${getErrorMessage(error)}` } : message
      )));
    } finally {
      setIsLoading(false);
    }
  };

  const handleHistoryAction = async (action: 'show' | 'reset', provider: string, sessionId: string) => {
    if (action === 'show') {
      try {
        const historyData = await getHistory(provider, sessionId);
        const historyContent = formatHistoryMessage(provider, sessionId, historyData.turns || []);
        setMessages(prev => [...prev, { id: createMessageId(), role: 'system', content: historyContent }]);
      } catch (error) {
        setMessages(prev => [...prev, { id: createMessageId(), role: 'system', content: `Error getting history: ${getErrorMessage(error)}` }]);
      }
      return;
    }

    try {
      await resetMemory(provider, sessionId);
      setMessages(prev => [...prev, { id: createMessageId(), role: 'system', content: `✅ Memory cleared for ${provider} (${sessionId})` }]);
    } catch (error) {
      setMessages(prev => [...prev, { id: createMessageId(), role: 'system', content: `Error resetting memory: ${getErrorMessage(error)}` }]);
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <FrameworkHeader
        title="LangGraph UI (LangGraph SDK + Gateway)"
        color="blue"
        settings={settings}
        onSettingsClick={() => setShowSettings(!showSettings)}
        apiStatus={apiStatus}
        apiCapabilities={apiCapabilities}
      />

      <div className="flex-1 overflow-hidden p-4">
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
                  value={input || ''}
                  onChange={(e) => setInput(e.target.value)}
                  type="text"
                  placeholder={`Ask questions... (${settings.queryMode === 'single' ? settings.selectedProvider : 'all providers'})`}
                  className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  disabled={isLoading || !input?.trim()}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-md"
                >
                  Send
                </button>
              </div>
            </form>
            <div className="text-xs text-gray-500 mt-2">
              Powered by LangGraph SDK client + gateway • {LANGGRAPH_API_URL} • assistant: {LANGGRAPH_ASSISTANT_ID} • client: {langGraphClient ? 'ready' : 'n/a'}
            </div>
          </div>
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

// LlamaIndex Component

export default LangGraphChatPage;
