'use client';

import { useMemo, useState } from 'react';
import { ChatInput, ChatSection, type Message as LlamaMessage } from '@llamaindex/chat-ui';
import {
  ResponseDetails,
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
  parseMessageEnvelope,
  extractDetailsFromContent,
  createMessageId,
  LANGGRAPH_API_URL
} from './shared';

const toMessageText = (message: LlamaMessage): string => {
  const textPart = message.parts?.find((part: any) => part?.type === 'text') as any;
  const rawText = textPart?.text || '';
  return parseMessageEnvelope(rawText)?.content || rawText;
};

const toMessageDetails = (message: LlamaMessage): ResponseDetails | undefined => {
  const textPart = message.parts?.find((part: any) => part?.type === 'text') as any;
  const rawText = textPart?.text || '';
  const envelope = parseMessageEnvelope(rawText);
  return envelope?.details || extractDetailsFromContent(rawText);
};

const createTextMessage = (role: LlamaMessage['role'], text: string, id = String(createMessageId())): LlamaMessage => ({
  id,
  role,
  parts: [{ type: 'text', text }]
});

const LlamaIndexChatPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const { providers, settings, setSettings, apiStatus, checkApiStatus, apiCapabilities } = useAPISettings();

  const [messages, setMessages] = useState<LlamaMessage[]>([
    createTextMessage('assistant', 'Hello! I\'m your LlamaIndex assistant connected to your API.', 'welcome')
  ]);
  const [status, setStatus] = useState<'submitted' | 'streaming' | 'ready' | 'error'>('ready');
  const [detailsById, setDetailsById] = useState<Record<string, ResponseDetails>>({});
  const [activeCoagentCalls, setActiveCoagentCalls] = useState<CoagentCallDetails[]>([]);

  const appendSystemMessage = (content: string) => {
    setMessages((prev) => [...prev, createTextMessage('system', content)]);
  };

  const sendMessage = async (message: LlamaMessage) => {
    const userText = toMessageText(message).trim();
    if (!userText || status === 'submitted' || status === 'streaming') return;

    const assistantId = String(createMessageId());
    const assistantMessage = createTextMessage('assistant', '', assistantId);

    setMessages((prev) => [...prev, createTextMessage('user', userText, message.id), assistantMessage]);
    setStatus(settings.responseMode === 'stream' ? 'streaming' : 'submitted');

    try {
      const result = await callAPI(userText, settings, {
        onChunk: (partialText) => {
          setMessages((prev) => prev.map((m) => (m.id === assistantId ? createTextMessage('assistant', partialText, assistantId) : m)));
        }
      });

      setMessages((prev) => prev.map((m) => (m.id === assistantId ? createTextMessage('assistant', result.content, assistantId) : m)));
      if (result.details) {
        setDetailsById((prev) => ({ ...prev, [assistantId]: result.details! }));
        setActiveCoagentCalls(result.details.coagentCalls || []);
      }
      setStatus('ready');
    } catch (error) {
      setMessages((prev) => prev.map((m) => (
        m.id === assistantId ? createTextMessage('assistant', `Connection Error: ${getErrorMessage(error)}`, assistantId) : m
      )));
      setStatus('error');
    }
  };

  const handler = useMemo(
    () => ({
      messages,
      status,
      sendMessage,
      setMessages
    }),
    [messages, status]
  );

  const handleHistoryAction = async (action: 'show' | 'reset', provider: string, sessionId: string) => {
    if (action === 'show') {
      try {
        const historyData = await getHistory(provider, sessionId);
        appendSystemMessage(formatHistoryMessage(provider, sessionId, historyData.turns || []));
      } catch (error) {
        appendSystemMessage(`Error getting history: ${getErrorMessage(error)}`);
      }
      return;
    }

    try {
      await resetMemory(provider, sessionId);
      appendSystemMessage(`✅ Memory cleared for ${provider} (${sessionId})`);
    } catch (error) {
      appendSystemMessage(`Error resetting memory: ${getErrorMessage(error)}`);
    }
  };

  const isLlamaLoading = status === 'submitted' || status === 'streaming';

  return (
    <div className="h-screen flex flex-col">
      <FrameworkHeader
        title="LlamaIndex Chat UI (@llamaindex/chat-ui, in-component adapter)"
        color="purple"
        settings={settings}
        onSettingsClick={() => setShowSettings(!showSettings)}
        apiStatus={apiStatus}
        apiCapabilities={apiCapabilities}
      />

      <div className="flex-1 overflow-hidden p-4">
        <div className={`grid h-full gap-4 ${apiCapabilities.hasCoagent ? 'lg:grid-cols-[minmax(0,1fr)_24rem]' : ''}`}>
          <div className="h-full bg-white rounded-lg border overflow-hidden flex flex-col">
          <ChatSection handler={handler} className="h-full flex flex-col">
            <div className="flex-1 overflow-y-auto p-4">
              {messages.map((message) => {
                const text = toMessageText(message);
                const isAssistantPending = message.role === 'assistant' && !text.trim() && isLlamaLoading;

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
                      {message.role === 'assistant' && (
                        <ResponseDetailsPanel details={detailsById[message.id] || toMessageDetails(message)} />
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            <ChatInput className="border-t">
              <ChatInput.Form className="p-4">
                <ChatInput.Field
                  placeholder={`Ask questions... (${settings.queryMode === 'single' ? settings.selectedProvider : 'all providers'})`}
                  className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
                <div className="mt-2 flex items-center justify-between">
                  <ChatInput.Submit
                    disabled={isLlamaLoading}
                    className="ml-2 rounded-lg bg-purple-600 px-6 py-2 text-white hover:bg-purple-700 disabled:bg-gray-400"
                  >
                    {isLlamaLoading ? 'Generating...' : 'Send'}
                  </ChatInput.Submit>
                </div>
              </ChatInput.Form>
              <div className="text-xs text-gray-500">
                Powered by @llamaindex/chat-ui (no local API route) • application: {LANGGRAPH_API_URL}
                {' • '}
                {apiCapabilities.hasMemory ? `Session: ${settings.sessionId}` : 'No memory'}
                {isLlamaLoading && <span className="ml-2">• Generating...</span>}
               </div>	    	      
            </ChatInput>
          </ChatSection>
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

      {showSettings && <div className="fixed inset-0 z-40 bg-black bg-opacity-50" onClick={() => setShowSettings(false)} />}
    </div>
  );
};

export default LlamaIndexChatPage;
