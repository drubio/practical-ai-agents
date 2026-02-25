'use client';

import React, { useState } from 'react';
import {
  ThreadPrimitive,
  MessagePrimitive,
  ComposerPrimitive,
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
  useMessage
} from '@assistant-ui/react';
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

const MaterialUIChatPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const [isAssistantLoading, setIsAssistantLoading] = useState(false);
  const [assistantNotices, setAssistantNotices] = useState<string[]>([]);
  const { providers, settings, setSettings, apiStatus, checkApiStatus, apiCapabilities } = useAPISettings();

  const appendAssistantNotice = (content: string) => {
    setAssistantNotices((prev) => [...prev, content]);
  };

  // Assistant UI adapter for your API
  const modelAdapter: ChatModelAdapter = {
    async run({ messages, abortSignal }) {
      const lastMessage = messages[messages.length - 1];
      
      setIsAssistantLoading(true);
      try {
        // Extract text content properly from the message
        let messageText = '';
        if (lastMessage.content) {
          if (Array.isArray(lastMessage.content)) {
            // Handle content array format
            const textContent = lastMessage.content.find(c => c.type === 'text');
            messageText = textContent ? textContent.text : '';
          } else if (typeof lastMessage.content === 'string') {
            // Handle string format
            messageText = lastMessage.content;
          }
        }

        if (!messageText) {
          throw new Error('No text content found in message');
        }

        if (abortSignal?.aborted) {
          throw new Error('Request aborted');
        }

        const result = await callAPI(messageText, settings);

        console.log('Assistant UI API response:', result);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(result)
            }
          ]
        };
      } catch (error) {
        console.error('Assistant UI API error:', error);
        return {
          content: [
            {
              type: "text", 
              text: `Connection Error: ${getErrorMessage(error)}`
            }
          ]
        };
      } finally {
        setIsAssistantLoading(false);
      }
    }
  };

  // Assistant UI runtime using useLocalRuntime
  const runtime = useLocalRuntime(modelAdapter);

  const handleHistoryAction = async (action: 'show' | 'reset', provider: string, sessionId: string) => {
    if (action === 'show') {
      try {
        const historyData = await getHistory(provider, sessionId);
        const historyContent = formatHistoryMessage(provider, sessionId, historyData.turns || []);

        appendAssistantNotice(historyContent);
      } catch (error) {
        appendAssistantNotice(`Error getting history: ${getErrorMessage(error)}`);
      }
    } else if (action === 'reset') {
      try {
        await resetMemory(provider, sessionId);
        appendAssistantNotice(`✅ Memory cleared for ${provider} (${sessionId})`);
      } catch (error) {
        appendAssistantNotice(`Error resetting memory: ${getErrorMessage(error)}`);
      }
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <FrameworkHeader 
        title="Material UI (@assistant-ui/react)"
        color="green"
        settings={settings}
        onSettingsClick={() => setShowSettings(!showSettings)}
        apiStatus={apiStatus}
        apiCapabilities={apiCapabilities}
      />
      
      <div className="flex-1 overflow-hidden p-4">
        <div className="h-full bg-white rounded-lg border overflow-hidden flex flex-col">
          {/* Assistant UI Components */}
          <AssistantRuntimeProvider runtime={runtime}>
            <ThreadPrimitive.Root className="h-full bg-gradient-to-b from-green-50 to-white flex flex-col">
              <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto p-4">
                {assistantNotices.map((notice, index) => (
                  <div key={`${index}-${notice.slice(0, 24)}`} className="mb-4 text-left">
                    <div className="bg-emerald-50 border border-emerald-200 text-emerald-900 px-4 py-2 rounded-lg max-w-2xl whitespace-pre-wrap">
                      {notice}
                    </div>
                  </div>
                ))}
                <ThreadPrimitive.Empty>
                  <div className="text-center text-gray-500 mt-8">
                    Hello! I'm your Material UI chat layer connected to your API.
                  </div>
                </ThreadPrimitive.Empty>
                <ThreadPrimitive.Messages 
                  components={{
                    UserMessage: () => (
                      <div className="mb-4 text-right">
                        <div className="bg-green-600 text-white px-4 py-2 rounded-lg inline-block max-w-md">
                          <MessagePrimitive.Content />
                        </div>
                      </div>
                    ),
                    AssistantMessage: () => {
                      const rawText = useMessage((state: any) => state.content)
                        ?.find?.((part: any) => part?.type === 'text')?.text || '';
                      const envelope = parseMessageEnvelope(rawText);
                      const text = envelope?.content || rawText;
                      const details = envelope?.details || extractDetailsFromContent(rawText);

                      return (
                        <div className="mb-4 text-left">
                          <div className="bg-gray-100 text-gray-900 px-4 py-2 rounded-lg inline-block max-w-md whitespace-pre-wrap">
                            <div>{!text?.trim() && isAssistantLoading ? <ThinkingIndicator label="Generating response..." /> : text}</div>
                            <ResponseDetailsPanel details={details} />
                          </div>
                        </div>
                      );
                    }
                  }}
                />
              </ThreadPrimitive.Viewport>
              <div className="border-t p-4">
                <ComposerPrimitive.Root>
                  <div className="flex space-x-2">
                    <ComposerPrimitive.Input 
                      placeholder={`Ask questions... ${apiCapabilities.hasMemory ? `(Session: ${settings.sessionId})` : ''}`}
                      className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-500"
                    />
                    <ComposerPrimitive.Send
                      disabled={isAssistantLoading}
                      className="bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg"
                    >
                      {isAssistantLoading ? 'Generating...' : 'Send'}
                    </ComposerPrimitive.Send>
                  </div>
                  <div className="text-xs text-gray-500 mt-2">
                    Powered by @assistant-ui/react • Real streaming conversations
                    {isAssistantLoading && <span className="ml-2">• Generating...</span>}
                  </div>
                </ComposerPrimitive.Root>
              </div>
            </ThreadPrimitive.Root>
          </AssistantRuntimeProvider>
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

// Custom Chat Component (vanilla React implementation)

export default MaterialUIChatPage;
