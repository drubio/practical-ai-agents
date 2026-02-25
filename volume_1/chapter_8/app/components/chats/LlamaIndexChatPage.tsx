'use client';

import React, { useState } from 'react';
import { ChatSection, ChatInput } from '@llamaindex/chat-ui';
import { useChat } from '@ai-sdk/react';
import { TextStreamChatTransport } from 'ai';
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
  extractDetailsFromContent,
  LANGGRAPH_API_URL,
  LANGGRAPH_ASSISTANT_ID
} from './shared';

const LlamaIndexChatPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const { providers, settings, setSettings, apiStatus, checkApiStatus, apiCapabilities } = useAPISettings();

  const [llamaResponseDetails, setLlamaResponseDetails] = useState<Record<string, ResponseDetails>>({});

  const chat = useChat<any>({
    transport: new TextStreamChatTransport({
      api: '/api/llamaindex-chat',
      body: {
        queryMode: settings.queryMode,
        selectedProvider: settings.selectedProvider,
        temperature: settings.temperature,
        maxTokens: settings.maxTokens,
        sessionId: settings.sessionId,
        responseMode: settings.responseMode,
        template: '{topic}'
      }
    }) as any,
    onError: (error) => console.error('LlamaIndex chat error:', error),
    onFinish: ({ message }: any) => {
      const text = message?.parts?.find((part: any) => part?.type === 'text')?.text;
      if (!text || typeof text !== 'string') return;
      const envelope = parseMessageEnvelope(text);
      const details = envelope?.details || extractDetailsFromContent(text);
      if (!details) return;
      setLlamaResponseDetails((prev) => ({ ...prev, [message.id]: details }));
    },
    messages: [
      {
        id: 'welcome',
        role: 'assistant',
        parts: [{ type: 'text', text: 'Hello! I\'m your LlamaIndex assistant connected to your API.' }]
      }
    ]
  });

  const isLlamaLoading = chat.status === 'submitted' || chat.status === 'streaming';
  const hasPendingLlamaAssistantMessage = chat.messages.some((message: any) => {
    if (message?.role !== 'assistant') return false;
    const rawText = message?.parts?.find((part: any) => part?.type === 'text')?.text || '';
    const envelope = parseMessageEnvelope(rawText);
    const text = envelope?.content || rawText;
    return !text?.trim();
  });

  const appendSystemMessage = (content: string) => {
    chat.setMessages((messages) => [
      ...messages,
      {
        id: `system-${Date.now()}`,
        role: 'system',
        parts: [{ type: 'text', text: content }]
      }
    ]);
  };

  const handleHistoryAction = async (action: 'show' | 'reset', provider: string, sessionId: string) => {
    if (action === 'show') {
      try {
        const historyData = await getHistory(provider, sessionId);
        const historyContent = formatHistoryMessage(provider, sessionId, historyData.turns || []);
        
        // Add history to LlamaIndex chat
        appendSystemMessage(historyContent);
      } catch (error) {
        appendSystemMessage(`Error getting history: ${getErrorMessage(error)}`);
      }
    } else if (action === 'reset') {
      try {
        await resetMemory(provider, sessionId);
        appendSystemMessage(`✅ Memory cleared for ${provider} (${sessionId})`);
      } catch (error) {
        appendSystemMessage(`Error resetting memory: ${getErrorMessage(error)}`);
      }
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <FrameworkHeader 
        title="LlamaIndex Chat UI (Real Components)"
        color="purple"
        settings={settings}
        onSettingsClick={() => setShowSettings(!showSettings)}
        apiStatus={apiStatus}
        apiCapabilities={apiCapabilities}
      />
      
      <div className="flex-1 overflow-hidden p-4">
        <div className="h-full bg-white rounded-lg border overflow-hidden flex flex-col">
          <ChatSection handler={chat} className="h-full flex flex-col">
            <div className="flex-1 overflow-y-auto p-4">
              {chat.messages.map((message: any) => {
                const rawText = message?.parts?.find((part: any) => part?.type === 'text')?.text || '';
                const envelope = parseMessageEnvelope(rawText);
                const text = envelope?.content || rawText;
                const isAssistantPending = message.role === 'assistant' && !text?.trim() && chat.status !== 'ready';
                return (
                  <div key={message.id} className={`mb-4 ${message.role === 'user' ? 'text-right' : message.role === 'system' ? 'text-center' : 'text-left'}`}>
                    <div className={`inline-block px-4 py-2 rounded-lg ${
                      message.role === 'user'
                        ? 'bg-purple-600 text-white max-w-md'
                        : message.role === 'system'
                        ? 'bg-amber-50 text-amber-900 border border-amber-200 max-w-3xl text-left whitespace-pre-wrap'
                        : 'bg-purple-100 text-gray-900 max-w-md whitespace-pre-wrap'
                    }`}>
                      {isAssistantPending ? <ThinkingIndicator label="Generating response..." /> : text}
                      {message.role === 'assistant' && <ResponseDetailsPanel details={llamaResponseDetails[message.id]} />}
                    </div>
                  </div>
                );
              })}
              {isLlamaLoading && !hasPendingLlamaAssistantMessage && (
                <div className="mb-4 text-left">
                  <div className="inline-block rounded-lg bg-purple-100 px-4 py-2 text-gray-900">
                    <ThinkingIndicator label="Generating response..." />
                  </div>
                </div>
              )}
            </div>
            <ChatInput className="border-t">
              <ChatInput.Form className="p-4">
                <ChatInput.Field 
                  placeholder={`Ask questions... (${settings.queryMode === 'single' ? settings.selectedProvider : 'all providers'})`}
                  className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
                <div className="flex justify-between items-center mt-2">
                  <div className="text-xs text-gray-500">
                    Powered by @llamaindex/chat-ui • {apiCapabilities.hasMemory ? `Session: ${settings.sessionId}` : 'No memory'}
                    {isLlamaLoading && <span className="ml-2">• Generating...</span>}
                  </div>
                  <ChatInput.Submit
                    disabled={isLlamaLoading}
                    className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg ml-2"
                  >
                    {isLlamaLoading ? 'Generating...' : 'Send'}
                  </ChatInput.Submit>
                </div>
              </ChatInput.Form>
            </ChatInput>
          </ChatSection>
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

// Assistant UI Component

export default LlamaIndexChatPage;
