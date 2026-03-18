'use client';

import { useMemo, useState } from 'react';
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  type ChatModelAdapter,
  useLocalRuntime,
  useMessage
} from '@assistant-ui/react';
import {
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
  extractDetailsFromContent
} from './shared';

const getAssistantMessageText = (content: unknown): string => {
  if (typeof content === 'string') return content.trim();
  if (!Array.isArray(content)) return '';

  return content
    .map((part: any) => (part?.type === 'text' && typeof part?.text === 'string' ? part.text : ''))
    .join('')
    .trim();
};

const AssistantMessage = ({ isAssistantLoading }: { isAssistantLoading: boolean }) => {
  const rawText = useMessage((state: any) => state.content)
    ?.find?.((part: any) => part?.type === 'text')?.text || '';
  const envelope = parseMessageEnvelope(rawText);
  const text = envelope?.content || rawText;
  const details = envelope?.details || extractDetailsFromContent(rawText);

  return (
    <div className="mb-4 text-left">
      <div className="inline-block max-w-md whitespace-pre-wrap rounded-lg bg-gray-100 px-4 py-2 text-gray-900">
        <div>{!text.trim() && isAssistantLoading ? <ThinkingIndicator label="Generating response..." /> : text}</div>
        <ResponseDetailsPanel details={details} />
      </div>
    </div>
  );
};

const AssistantUIChatPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const [isAssistantLoading, setIsAssistantLoading] = useState(false);
  const [assistantNotices, setAssistantNotices] = useState<string[]>([]);
  const [activeCoagentCalls, setActiveCoagentCalls] = useState<CoagentCallDetails[]>([]);
  const { providers, settings, setSettings, apiStatus, checkApiStatus, apiCapabilities } = useAPISettings();

  const appendAssistantNotice = (content: string) => {
    setAssistantNotices((prev) => [...prev, content]);
  };

  const modelAdapter: ChatModelAdapter = useMemo(
    () => ({
      async *run({ messages, abortSignal }) {
        const lastMessage = messages[messages.length - 1];
        const messageText = getAssistantMessageText(lastMessage?.content);

        if (!messageText) {
          yield { content: [{ type: 'text', text: 'Connection Error: No text content found in message' }] };
          return;
        }

        if (abortSignal?.aborted) {
          yield { content: [{ type: 'text', text: 'Connection Error: Request aborted' }] };
          return;
        }

        setIsAssistantLoading(true);
        try {
          if (settings.responseMode === 'stream') {
            let done = false;
            let finalPayload = '';
            let streamError: unknown;
            const queue: string[] = [];
            let wake: (() => void) | null = null;

            const push = (text: string) => {
              queue.push(text);
              if (wake) {
                wake();
                wake = null;
              }
            };

            void callAPI(messageText, settings, {
              onChunk: (partialText) => {
                push(partialText);
              }
            })
              .then((response) => {
                setActiveCoagentCalls(response.details?.coagentCalls || []);
                finalPayload = JSON.stringify(response);
                done = true;
                if (finalPayload.trim()) push(finalPayload);
                if (wake) {
                  wake();
                  wake = null;
                }
              })
              .catch((error) => {
                streamError = error;
                done = true;
                if (wake) {
                  wake();
                  wake = null;
                }
              });

            while (!done || queue.length > 0) {
              if (queue.length === 0) {
                await new Promise<void>((resolve) => {
                  wake = resolve;
                });
                continue;
              }

              const nextText = queue.shift();
              if (!nextText) continue;
              yield { content: [{ type: 'text', text: nextText }] };
            }

            if (streamError) {
              throw streamError;
            }

            return;
          }

          const result = await callAPI(messageText, settings);
          setActiveCoagentCalls(result.details?.coagentCalls || []);
          yield { content: [{ type: 'text', text: JSON.stringify(result) }] };
        } catch (error) {
          yield {
            content: [{ type: 'text', text: `Connection Error: ${getErrorMessage(error)}` }]
          };
        } finally {
          setIsAssistantLoading(false);
        }
      }
    }),
    [settings]
  );

  const runtime = useLocalRuntime(modelAdapter);

  const handleHistoryAction = async (action: 'show' | 'reset', provider: string, sessionId: string) => {
    if (action === 'show') {
      try {
        const historyData = await getHistory(provider, sessionId);
        appendAssistantNotice(formatHistoryMessage(provider, sessionId, historyData.turns || []));
      } catch (error) {
        appendAssistantNotice(`Error getting history: ${getErrorMessage(error)}`);
      }
      return;
    }

    try {
      await resetMemory(provider, sessionId);
      appendAssistantNotice(`✅ Memory cleared for ${provider} (${sessionId})`);
    } catch (error) {
      appendAssistantNotice(`Error resetting memory: ${getErrorMessage(error)}`);
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <FrameworkHeader
        title="Assistant UI (@assistant-ui/react)"
        color="green"
        settings={settings}
        onSettingsClick={() => setShowSettings(!showSettings)}
        apiStatus={apiStatus}
        apiCapabilities={apiCapabilities}
      />

      <div className="flex-1 overflow-hidden p-4">
        <div className={`grid h-full gap-4 ${apiCapabilities.hasCoagent ? 'lg:grid-cols-[minmax(0,1fr)_24rem]' : ''}`}>
          <div className="h-full overflow-hidden rounded-lg border bg-white">
          <AssistantRuntimeProvider runtime={runtime}>
            <ThreadPrimitive.Root className="flex h-full flex-col bg-gradient-to-b from-green-50 to-white">
              <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto p-4">
                {assistantNotices.map((notice, index) => (
                  <div key={`${index}-${notice.slice(0, 24)}`} className="mb-4 text-left">
                    <div className="max-w-2xl whitespace-pre-wrap rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-2 text-emerald-900">
                      {notice}
                    </div>
                  </div>
                ))}

                <ThreadPrimitive.Empty>
                  <div className="mt-8 text-center text-gray-500">Hello! I'm your Assistant UI chat layer connected to your API.</div>
                </ThreadPrimitive.Empty>

                <ThreadPrimitive.Messages
                  components={{
                    UserMessage: () => (
                      <div className="mb-4 text-right">
                        <div className="inline-block max-w-md rounded-lg bg-green-600 px-4 py-2 text-white">
                          <MessagePrimitive.Content />
                        </div>
                      </div>
                    ),
                    AssistantMessage: () => <AssistantMessage isAssistantLoading={isAssistantLoading} />
                  }}
                />
              </ThreadPrimitive.Viewport>

              <div className="border-t p-4">
                <ComposerPrimitive.Root>
                  <div className="flex space-x-2">
                    <ComposerPrimitive.Input
                      placeholder={`Ask questions... ${apiCapabilities.hasMemory ? `(Session: ${settings.sessionId})` : ''}`}
                      className="flex-1 rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-500"
                    />
                    <ComposerPrimitive.Send
                      disabled={isAssistantLoading}
                      className="rounded-lg bg-green-600 px-6 py-2 text-white hover:bg-green-700 disabled:bg-gray-400"
                    >
                      {isAssistantLoading ? 'Generating...' : 'Send'}
                    </ComposerPrimitive.Send>
                  </div>
                  <div className="mt-2 text-xs text-gray-500">
                    Powered by @assistant-ui/react runtime adapter • mode: {settings.responseMode}
                    {isAssistantLoading && <span className="ml-2">• Generating...</span>}
                  </div>
                </ComposerPrimitive.Root>
              </div>
            </ThreadPrimitive.Root>
          </AssistantRuntimeProvider>
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

export default AssistantUIChatPage;
