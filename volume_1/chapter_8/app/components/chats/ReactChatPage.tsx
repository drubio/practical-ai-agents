'use client';

import { useCallback, useMemo, useReducer, useState } from 'react';
import {
  ChatMessage,
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
  createMessageId
} from './shared';

type ChatState = {
  messages: ChatMessage[];
  isLoading: boolean;
};

type ChatAction =
  | { type: 'start'; userContent: string; assistantId: number }
  | { type: 'assistant-update'; assistantId: number; content: string; details?: ChatMessage['details'] }
  | { type: 'append-system'; content: string }
  | { type: 'finish' };

const initialState: ChatState = {
  messages: [
    { id: 1, role: 'assistant', content: 'Hello! I\'m your custom chat assistant connected to your API.' }
  ],
  isLoading: false
};

const chatReducer = (state: ChatState, action: ChatAction): ChatState => {
  switch (action.type) {
    case 'start': {
      return {
        ...state,
        isLoading: true,
        messages: [
          ...state.messages,
          { id: createMessageId(), role: 'user', content: action.userContent },
          { id: action.assistantId, role: 'assistant', content: '' }
        ]
      };
    }

    case 'assistant-update': {
      return {
        ...state,
        messages: state.messages.map((message) => (
          message.id === action.assistantId
            ? { ...message, content: action.content, details: action.details ?? message.details }
            : message
        ))
      };
    }

    case 'append-system': {
      return {
        ...state,
        messages: [...state.messages, { id: createMessageId(), role: 'system', content: action.content }]
      };
    }

    case 'finish': {
      return { ...state, isLoading: false };
    }

    default:
      return state;
  }
};

const ReactChatPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const { providers, settings, setSettings, apiStatus, checkApiStatus, apiCapabilities } = useAPISettings();
  const [input, setInput] = useState('');
  const [{ messages, isLoading }, dispatch] = useReducer(chatReducer, initialState);

  const formatProviderAggregateText = useCallback(
    (text: string) => (settings.queryMode === 'all' ? text.replace('Results from', 'Custom Chat Results from') : text),
    [settings.queryMode]
  );

  const sessionHint = useMemo(
    () => (apiCapabilities.hasMemory ? `(Session: ${settings.sessionId})` : '(No memory)'),
    [apiCapabilities.hasMemory, settings.sessionId]
  );

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const userInput = input.trim();
    if (!userInput || isLoading) return;

    const assistantId = createMessageId();
    dispatch({ type: 'start', userContent: userInput, assistantId });
    setInput('');

    try {
      const response = await callAPI(userInput, settings, {
        onChunk: (partialText) => {
          dispatch({
            type: 'assistant-update',
            assistantId,
            content: formatProviderAggregateText(partialText)
          });
        }
      });

      dispatch({
        type: 'assistant-update',
        assistantId,
        content: formatProviderAggregateText(response.content),
        details: response.details
      });
    } catch (error) {
      dispatch({
        type: 'assistant-update',
        assistantId,
        content: `Connection Error: ${getErrorMessage(error)}`
      });
    } finally {
      dispatch({ type: 'finish' });
    }
  };

  const handleHistoryAction = async (action: 'show' | 'reset', provider: string, sessionId: string) => {
    if (action === 'show') {
      try {
        const historyData = await getHistory(provider, sessionId);
        dispatch({
          type: 'append-system',
          content: formatHistoryMessage(provider, sessionId, historyData.turns || [])
        });
      } catch (error) {
        dispatch({ type: 'append-system', content: `Error getting history: ${getErrorMessage(error)}` });
      }
      return;
    }

    try {
      await resetMemory(provider, sessionId);
      dispatch({ type: 'append-system', content: `✅ Memory cleared for ${provider} (${sessionId})` });
    } catch (error) {
      dispatch({ type: 'append-system', content: `Error resetting memory: ${getErrorMessage(error)}` });
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <FrameworkHeader
        title="React UI (Vanilla React + useReducer)"
        color="orange"
        settings={settings}
        onSettingsClick={() => setShowSettings(!showSettings)}
        apiStatus={apiStatus}
        apiCapabilities={apiCapabilities}
      />

      <div className="flex-1 overflow-hidden p-4">
        <div className="h-full bg-gradient-to-b from-orange-50 to-white rounded-lg border overflow-hidden flex flex-col">
          <div className="flex-1 overflow-y-auto p-4">
            {messages.map((message) => (
              <div key={message.id} className={`mb-4 ${message.role === 'user' ? 'text-right' : message.role === 'system' ? 'text-center' : 'text-left'}`}>
                <div className={`inline-block px-4 py-2 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-orange-600 text-white max-w-md'
                    : message.role === 'system'
                    ? 'bg-amber-50 text-amber-900 border border-amber-200 max-w-3xl text-left whitespace-pre-wrap'
                    : 'bg-orange-100 text-gray-900 max-w-md whitespace-pre-wrap'
                }`}>
                  {message.role === 'assistant' && <div className="mb-1 text-xs font-semibold text-orange-700">Custom Assistant</div>}
                  {message.role === 'system' && (
                    <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-amber-700">Conversation History</div>
                  )}
                  {message.role === 'assistant' && !message.content && isLoading ? (
                    <ThinkingIndicator label="Processing your request..." />
                  ) : (
                    message.content
                  )}
                  {message.role === 'assistant' && <ResponseDetailsPanel details={message.details} />}
                </div>
              </div>
            ))}
          </div>

          <div className="border-t bg-white p-4">
            <form onSubmit={handleSubmit}>
              <div className="flex space-x-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={`Ask questions... ${sessionHint}`}
                  disabled={isLoading}
                  className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-1 focus:ring-orange-500"
                />
                <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="bg-orange-600 hover:bg-orange-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg"
                >
                  Send
                </button>
              </div>
            </form>
            <div className="text-xs text-gray-500 mt-2">
              Custom React chat interface • React primitives: useReducer + useMemo + useCallback
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

export default ReactChatPage;
