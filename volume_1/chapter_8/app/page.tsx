'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { Settings } from 'lucide-react';

// Framework imports
import { Client as LangGraphClient } from '@langchain/langgraph-sdk';
import { ChatSection, ChatMessages, ChatInput } from '@llamaindex/chat-ui';
import { useChat } from '@ai-sdk/react';
import { TextStreamChatTransport } from 'ai';
import { 
  ThreadPrimitive, 
  MessagePrimitive, 
  ComposerPrimitive,
  AssistantRuntimeProvider, 
  useLocalRuntime,
  type ChatModelAdapter
} from '@assistant-ui/react';

type ChatRole = 'assistant' | 'user' | 'system';
type QueryMode = 'single' | 'all';
type ResponseMode = 'stream' | 'standard';
type APIStatus = 'online' | 'offline' | 'checking';

type Provider = {
  name: string;
  display_name: string;
};

type ChatMessage = {
  id: number;
  role: ChatRole;
  content: string;
};

type APISettings = {
  queryMode: QueryMode;
  selectedProvider: string;
  temperature: number;
  maxTokens: number;
  sessionId: string;
  responseMode: ResponseMode;
};

type APICapabilities = {
  hasMemory: boolean;
  hasHistory: boolean;
  framework: string;
  hasStreaming: boolean;
};

type CallAPIOptions = {
  onChunk?: (partialText: string) => void;
};

const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
};

const GATEWAY_API_BASE = 'http://localhost:8000';
const LANGGRAPH_API_URL = GATEWAY_API_BASE;
const LANGGRAPH_ASSISTANT_ID = 'agent';

// ============================================================================
// SHARED HOOKS AND UTILITIES
// ============================================================================

// Response processing utilities
const processApiResponse = (data: any, queryMode: string) => {
  if (queryMode === 'single') {
    if (data.success) {
      // Handle both old and new API response formats
      if (typeof data.response === 'string') {
        // Old API format - simple string response
        return data.response;
      } else if (typeof data.response === 'object' && data.response !== null) {
        // New API format - structured JSON response
        const structured = data.response;
        let content = '';
        
        if (structured.answer) {
          content += structured.answer;
        }
        
        if (structured.distilled && structured.distilled !== structured.answer) {
          content += `\n\n**Quick Answer**: ${structured.distilled}`;
        }
        
        if (structured.summary && structured.summary !== structured.answer) {
          content += `\n\n**Summary**: ${structured.summary}`;
        }
        
        if (structured.keywords && Array.isArray(structured.keywords) && structured.keywords.length > 0) {
          content += `\n\n**Keywords**: ${structured.keywords.join(', ')}`;
        }
        
        // Fallback to raw_answer if structured response is empty
        if (!content && data.raw_answer) {
          content = data.raw_answer;
        }
        
        return content || 'No response content available';
      } else {
        // Fallback to raw_answer for new API
        return data.raw_answer || 'No response available';
      }
    } else {
      return `Error: ${data.error}`;
    }
  } else {
    // Multi-provider mode
    if (data.success) {
      let content = `Results from ${data.summary?.total_providers || Object.keys(data.responses).length} providers:\n\n`;
      
      for (const [provider, response] of Object.entries(data.responses)) {
        const providerResponse = response as any;
        content += `**${provider}**: `;
        
        if (providerResponse.success) {
          if (typeof providerResponse.response === 'string') {
            // Old API format
            content += providerResponse.response;
          } else if (typeof providerResponse.response === 'object' && providerResponse.response !== null) {
            // New API format - use answer field or fallback to raw_answer
            const structured = providerResponse.response;
            content += structured.answer || providerResponse.raw_answer || 'No response available';
          } else {
            // Fallback
            content += providerResponse.raw_answer || 'No response available';
          }
        } else {
          content += `Error: ${providerResponse.error}`;
        }
        
        content += '\n\n';
      }
      
      return content;
    } else {
      return `Error: ${data.error}`;
    }
  }
};


const createMessageId = () => Date.now() + Math.floor(Math.random() * 100000);

const formatHistoryMessage = (provider: string, sessionId: string, turns: Array<{ role: string; content: string }> = []) => {
  if (!turns.length) {
    return `History for ${provider} (session: ${sessionId})\n\nNo conversation history found.`;
  }

  const lines = turns.map((turn, index) => {
    const roleLabel = turn.role === 'assistant' ? 'Assistant' : turn.role === 'user' ? 'User' : turn.role;
    return `${index + 1}. ${roleLabel}: ${turn.content}`;
  });

  return `History for ${provider} (session: ${sessionId})\n\n${lines.join('\n\n')}`;
};

// Custom hook for API settings and providers
const useAPISettings = () => {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [apiStatus, setApiStatus] = useState<APIStatus>('checking'); // 'online', 'offline', 'checking'
  const [apiCapabilities, setApiCapabilities] = useState<APICapabilities>({
    hasMemory: false,
    hasHistory: false,
    framework: '',
    hasStreaming: false
  });
  const [settings, setSettings] = useState<APISettings>({
    queryMode: 'single',
    selectedProvider: 'openai',
    temperature: 0.7,
    maxTokens: 1000,
    sessionId: 'default', // Added session ID support
    responseMode: 'stream' // stream | standard
  });

  const checkApiStatus = async () => {
    try {
      // Check main status endpoint
      const statusResponse = await fetch(`${GATEWAY_API_BASE}/`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      
      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        
        // Check providers endpoint
        const providersResponse = await fetch(`${GATEWAY_API_BASE}/providers`, {
          method: 'GET',
          signal: AbortSignal.timeout(5000)
        });
        
        if (providersResponse.ok) {
          const providersData = await providersResponse.json();
          setProviders(providersData.providers || []);
          setApiStatus('online');
          
          // Detect API capabilities
          let capabilitiesData = null;
          try {
            const capabilitiesResponse = await fetch(`${GATEWAY_API_BASE}/capabilities`, {
              method: 'GET',
              signal: AbortSignal.timeout(3000)
            });
            if (capabilitiesResponse.ok) {
              capabilitiesData = await capabilitiesResponse.json();
            }
          } catch (_) {
            capabilitiesData = null;
          }

          setApiCapabilities({
            hasMemory: statusData.framework?.includes('History') || capabilitiesData?.memory || false,
            hasHistory: statusData.framework?.includes('History') || capabilitiesData?.memory || false,
            framework: statusData.framework || providersData.framework || '',
            hasStreaming: Boolean(capabilitiesData?.streaming)
          });
          
          if (providersData.providers?.length > 0) {
            const providerNames = providersData.providers.map((provider: Provider) => provider.name);
            const defaultProvider = providerNames.includes('openai') ? 'openai' : providersData.providers[0].name;
            setSettings(prev => ({ ...prev, selectedProvider: defaultProvider }));
          }

          setSettings(prev => (
            prev.responseMode === 'stream' && !capabilitiesData?.streaming
              ? { ...prev, responseMode: 'standard' }
              : prev
          ));
        } else {
          setApiStatus('offline');
        }
      } else {
        setApiStatus('offline');
      }
    } catch (error) {
      setProviders([]);
      setApiStatus('offline');
      setApiCapabilities({ hasMemory: false, hasHistory: false, framework: '', hasStreaming: false });
      setSettings(prev => ({ ...prev, responseMode: 'standard' }));
    }
  };

  useEffect(() => {
    checkApiStatus();
    
    // Check API status every 30 seconds
    const interval = setInterval(checkApiStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  return { providers, settings, setSettings, apiStatus, checkApiStatus, apiCapabilities };
};

// Shared API call logic
const callAPI = async (message: string, settings: APISettings, options: CallAPIOptions = {}) => {
  const endpoint = settings.queryMode === 'single' ? '/query' : '/query-all';
  const payload = {
    topic: message,
    temperature: settings.temperature,
    max_tokens: settings.maxTokens,
    template: '{topic}',
    session_id: settings.sessionId,
    ...(settings.queryMode === 'single' && { provider: settings.selectedProvider })
  };

  const wantsStreaming = settings.responseMode === 'stream' && settings.queryMode === 'single';
  if (wantsStreaming && settings.queryMode === 'single') {
    const response = await fetch(`${GATEWAY_API_BASE}/query-stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok || !response.body) {
      throw new Error(`Streaming failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop() || '';

      for (const event of events) {
        const lines = event.split('\n').filter(line => line.startsWith('data: '));
        for (const line of lines) {
          const payloadLine = line.slice(6);
          if (!payloadLine) continue;
          const parsed = JSON.parse(payloadLine);

          if (parsed.type === 'chunk') {
            fullText += parsed.content || '';
            options.onChunk?.(fullText);
          } else if (parsed.type === 'error') {
            throw new Error(parsed.error || 'Streaming failed');
          }
        }
      }
    }

    return fullText;
  }

  const response = await fetch(`${GATEWAY_API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  const data = await response.json();
  return processApiResponse(data, settings.queryMode);
};

// History and memory management functions
const getHistory = async (provider: string, sessionId: string) => {
  try {
    const response = await fetch(`${GATEWAY_API_BASE}/history?provider=${provider}&session_id=${sessionId}`, {
      method: 'GET'
    });
    if (response.ok) {
      return await response.json();
    }
    throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    throw new Error(`Failed to get history: ${getErrorMessage(error)}`);
  }
};

const resetMemory = async (provider: string, sessionId: string) => {
  try {
    const response = await fetch(`${GATEWAY_API_BASE}/reset-memory`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider, session_id: sessionId })
    });
    if (response.ok) {
      return await response.json();
    }
    throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    throw new Error(`Failed to reset memory: ${getErrorMessage(error)}`);
  }
};

// API Status Indicator Component
const ApiStatusIndicator = ({ status, onRefresh }: { status: APIStatus; onRefresh: () => void }) => {
  const getStatusColor = () => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'offline': return 'bg-red-500';
      case 'checking': return 'bg-yellow-500 animate-pulse';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'online': return 'API Online';
      case 'offline': return 'API Offline';
      case 'checking': return 'Checking...';
      default: return 'Unknown';
    }
  };

  return (
    <div className="flex items-center justify-between p-3 bg-gray-50 border-b">
      <div className="flex items-center space-x-2">
        <div className={`w-3 h-3 rounded-full ${getStatusColor()}`}></div>
        <span className="text-sm font-medium">{getStatusText()}</span>
      </div>
      <button
        onClick={onRefresh}
        className="text-xs text-gray-600 hover:text-gray-800 underline"
      >
        Refresh
      </button>
    </div>
  );
};

// Shared Settings Sidebar Component
const SettingsSidebar = ({ isOpen, onClose, settings, onSettingsChange, providers, apiStatus, onRefreshApi, apiCapabilities, onHistoryAction }: {
  isOpen: boolean;
  onClose: () => void;
  settings: APISettings;
  onSettingsChange: React.Dispatch<React.SetStateAction<APISettings>>;
  providers: Provider[];
  apiStatus: APIStatus;
  onRefreshApi: () => void;
  apiCapabilities: APICapabilities;
  onHistoryAction: (action: 'show' | 'reset', provider: string, sessionId: string) => Promise<void>;
}) => (
  <div className={`fixed right-0 top-0 h-full w-80 bg-white border-l shadow-xl transform transition-transform z-50 ${
    isOpen ? 'translate-x-0' : 'translate-x-full'
  }`}>
    <div className="p-4 border-b bg-gray-50">
      <div className="flex justify-between items-center">
        <h2 className="text-lg font-semibold">API Settings</h2>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-700">✕</button>
      </div>
      {apiCapabilities.framework && (
        <div className="text-xs text-gray-600 mt-1">
          Framework: {apiCapabilities.framework}
        </div>
      )}
    </div>
    
    <ApiStatusIndicator status={apiStatus} onRefresh={onRefreshApi} />
    
    <div className="p-4 space-y-4 overflow-y-auto">
      <div>
        <label className="block text-sm font-medium mb-2">Query Mode</label>
        <div className="space-y-2">
          <label className="flex items-center">
            <input
              type="radio"
              value="single"
              checked={settings.queryMode === 'single'}
              onChange={(e) => onSettingsChange({ ...settings, queryMode: e.target.value as QueryMode })}
              className="text-blue-600"
            />
            <span className="ml-2 text-sm">Single Provider</span>
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              value="all"
              checked={settings.queryMode === 'all'}
              onChange={(e) => onSettingsChange({ ...settings, queryMode: e.target.value as QueryMode })}
              className="text-blue-600"
            />
            <span className="ml-2 text-sm">All Providers</span>
          </label>
        </div>
      </div>

      {settings.queryMode === 'single' && (
        <div>
          <label className="block text-sm font-medium mb-2">Provider</label>
          <select
            value={settings.selectedProvider}
            onChange={(e) => onSettingsChange({ ...settings, selectedProvider: e.target.value })}
            className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
            disabled={apiStatus !== 'online'}
          >
            {providers.map((p) => (
              <option key={p.name} value={p.name}>{p.display_name}</option>
            ))}
          </select>
        </div>
      )}

      {/* Session ID - only show if backend supports memory */}
      {apiCapabilities.hasMemory && (
        <div>
          <label className="block text-sm font-medium mb-2">Session ID</label>
          <input
            type="text"
            value={settings.sessionId}
            onChange={(e) => onSettingsChange({ ...settings, sessionId: e.target.value })}
            placeholder="default"
            className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Used for conversation memory
          </p>
          
          {/* History and Reset buttons - only show for single provider mode */}
          {settings.queryMode === 'single' && apiCapabilities.hasHistory && (
            <div className="flex space-x-2 mt-2">
              <button
                onClick={() => onHistoryAction('show', settings.selectedProvider, settings.sessionId)}
                className="flex-1 px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                disabled={apiStatus !== 'online' || !settings.selectedProvider}
              >
                Show History
              </button>
              <button
                onClick={() => onHistoryAction('reset', settings.selectedProvider, settings.sessionId)}
                className="flex-1 px-3 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200"
                disabled={apiStatus !== 'online' || !settings.selectedProvider}
              >
                Reset Memory
              </button>
            </div>
          )}
        </div>
      )}

      <div>
        <label className="block text-sm font-medium mb-2">Response Mode</label>
        <select
          value={settings.responseMode}
          onChange={(e) => onSettingsChange({ ...settings, responseMode: e.target.value as ResponseMode })}
          className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
        >
          <option value="stream" disabled={!apiCapabilities.hasStreaming}>
            Streaming {apiCapabilities.hasStreaming ? '' : '— unavailable'}
          </option>
          <option value="standard">Standard (wait for full response)</option>
        </select>
        <p className="text-xs text-gray-500 mt-1">
          {apiCapabilities.hasStreaming
            ? 'Backend supports streaming responses.'
            : 'Backend does not advertise streaming support.'}
        </p>
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Temperature: {settings.temperature}</label>
        <input
          type="range"
          min="0"
          max="2"
          step="0.1"
          value={settings.temperature}
          onChange={(e) => onSettingsChange({ ...settings, temperature: parseFloat(e.target.value) })}
          className="w-full"
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Max Tokens</label>
        <input
          type="number"
          min="1"
          max="4000"
          value={settings.maxTokens || ''}
          onChange={(e) => {
            const value = e.target.value;
            if (value === '' || (!isNaN(Number(value)) && Number(value) > 0)) {
              onSettingsChange({ 
                ...settings, 
                maxTokens: value === '' ? 1000 : parseInt(value, 10) 
              });
            }
          }}
          placeholder="1000"
          className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
        />
      </div>
    </div>
  </div>
);

// Shared Header Component
const FrameworkHeader = ({ title, color, settings, onSettingsClick, apiStatus, apiCapabilities }: {
  title: string;
  color: string;
  settings: APISettings;
  onSettingsClick: () => void;
  apiStatus: APIStatus;
  apiCapabilities: APICapabilities;
}) => (
  <div className={`bg-${color}-600 text-white p-3 flex justify-between items-center`}>
    <div className="flex items-center space-x-3">
      <div>
        <h1 className="text-lg font-semibold">{title}</h1>
        <div className="text-xs opacity-75">
          Mode: {settings.queryMode} | Provider: {settings.selectedProvider} | 
          {apiCapabilities.hasMemory && `Session: ${settings.sessionId} | `}
          Mode: {settings.responseMode} | Temp: {settings.temperature} | Max Tokens: {settings.maxTokens}
        </div>
      </div>
      <div className="flex items-center space-x-2">
        <div className={`w-2 h-2 rounded-full ${
          apiStatus === 'online' ? 'bg-green-400' : 
          apiStatus === 'offline' ? 'bg-red-400' : 
          'bg-yellow-400 animate-pulse'
        }`}></div>
        <span className="text-xs opacity-75">
          {apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Checking'}
        </span>
      </div>
    </div>
    <button 
      onClick={onSettingsClick}
      className={`p-2 hover:bg-${color}-700 rounded`}
    >
      <Settings className="w-4 h-4" />
    </button>
  </div>
);

// ============================================================================
// FRAMEWORK COMPONENTS
// ============================================================================

// LangChain Component
const LangChainPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const { providers, settings, setSettings, apiStatus, checkApiStatus, apiCapabilities } = useAPISettings();
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 1, role: 'assistant', content: 'Hello! I\'m your LangChain assistant connected to your gateway API.' }
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
      const content = await callAPI(input, settings, {
        onChunk: (partialText) => {
          setMessages(prev => prev.map((message) => (
            message.id === assistantId ? { ...message, content: partialText } : message
          )));
        }
      });
      setMessages(prev => prev.map((message) => (
        message.id === assistantId ? { ...message, content } : message
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
        title="LangChain Agent UI (LangGraph SDK + Gateway)"
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
                  {message.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="text-left mb-4">
                <div className="inline-block bg-gray-100 px-4 py-2 rounded-lg">
                  <div className="animate-pulse">LangChain agent processing...</div>
                </div>
              </div>
            )}
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
const LlamaIndexPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const { providers, settings, setSettings, apiStatus, checkApiStatus, apiCapabilities } = useAPISettings();

  const chat = useChat<any>({
    transport: new TextStreamChatTransport({
      api: '/api/llamaindex-agent',
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
    messages: [
      {
        id: 'welcome',
        role: 'assistant',
        parts: [{ type: 'text', text: 'Hello! I\'m your LlamaIndex assistant connected to your API.' }]
      }
    ]
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
            <ChatMessages className="flex-1 overflow-y-auto p-4" />
            <ChatInput className="border-t">
              <ChatInput.Form className="p-4">
                <ChatInput.Field 
                  placeholder={`Ask questions... (${settings.queryMode === 'single' ? settings.selectedProvider : 'all providers'})`}
                  className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
                <div className="flex justify-between items-center mt-2">
                  <div className="text-xs text-gray-500">
                    Powered by @llamaindex/chat-ui • {apiCapabilities.hasMemory ? `Session: ${settings.sessionId}` : 'No memory'}
                  </div>
                  <ChatInput.Submit className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg ml-2">
                    Send
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
const AssistantUIPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const { providers, settings, setSettings, apiStatus, checkApiStatus, apiCapabilities } = useAPISettings();

  // Assistant UI adapter for your API
  const modelAdapter: ChatModelAdapter = {
    async run({ messages, abortSignal }) {
      const lastMessage = messages[messages.length - 1];
      
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

        const endpoint = settings.queryMode === 'single' ? '/query' : '/query-all';
        const payload = {
          topic: messageText,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
          template: '{topic}',
          session_id: settings.sessionId,
          ...(settings.queryMode === 'single' && { provider: settings.selectedProvider })
        };

        console.log('Assistant UI API call:', { endpoint, payload });

        const response = await fetch(`${GATEWAY_API_BASE}${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: abortSignal
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        const content = processApiResponse(data, settings.queryMode);

        console.log('Assistant UI API response:', content);

        return {
          content: [
            {
              type: "text",
              text: content
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
        
        // Add system message to runtime - this might not work with current Assistant UI
        // For now, we'll show an alert or console log
        console.log('History:', historyContent);
        alert(`History loaded for ${provider} (${sessionId}). Check console for details.`);
      } catch (error) {
        console.error('Error getting history:', getErrorMessage(error));
        alert(`Error getting history: ${getErrorMessage(error)}`);
      }
    } else if (action === 'reset') {
      try {
        await resetMemory(provider, sessionId);
        alert(`✅ Memory cleared for ${provider} (${sessionId})`);
      } catch (error) {
        alert(`Error resetting memory: ${getErrorMessage(error)}`);
      }
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <FrameworkHeader 
        title="Assistant UI (Real Components)"
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
                <ThreadPrimitive.Empty>
                  <div className="text-center text-gray-500 mt-8">
                    Hello! I'm your Assistant UI connected to your API.
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
                    AssistantMessage: () => (
                      <div className="mb-4 text-left">
                        <div className="bg-gray-100 text-gray-900 px-4 py-2 rounded-lg inline-block max-w-md whitespace-pre-wrap">
                          <MessagePrimitive.Content />
                        </div>
                      </div>
                    )
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
                    <ComposerPrimitive.Send className="bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg">
                      Send
                    </ComposerPrimitive.Send>
                  </div>
                  <div className="text-xs text-gray-500 mt-2">
                    Powered by @assistant-ui/react • Real streaming conversations
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
const CustomChatPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const { providers, settings, setSettings, apiStatus, checkApiStatus, apiCapabilities } = useAPISettings();
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 1, role: 'assistant', content: 'Hello! I\'m your custom chat assistant connected to your API.' }
  ]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input?.trim() || isLoading) return;

    const userMessage: ChatMessage = { id: Date.now(), role: 'user', content: input };
    const assistantId = Date.now() + 1;
    setMessages(prev => [...prev, userMessage, { id: assistantId, role: 'assistant', content: '' }]);
    setIsLoading(true);
    setInput('');

    try {
      const content = await callAPI(input, settings, {
        onChunk: (partialText) => {
          const formattedPartial = settings.queryMode === 'all'
            ? partialText.replace('Results from', 'Custom Chat Results from')
            : partialText;
          setMessages(prev => prev.map(message => (
            message.id === assistantId ? { ...message, content: formattedPartial } : message
          )));
        }
      });

      const formattedContent = settings.queryMode === 'all' 
        ? content.replace('Results from', 'Custom Chat Results from')
        : content;
      
      setMessages(prev => prev.map(message => (
        message.id === assistantId ? { ...message, content: formattedContent } : message
      )));
    } catch (error) {
      setMessages(prev => prev.map(message => (
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
    } else if (action === 'reset') {
      try {
        await resetMemory(provider, sessionId);
        setMessages(prev => [...prev, { id: createMessageId(), role: 'system', content: `✅ Memory cleared for ${provider} (${sessionId})` }]);
      } catch (error) {
        setMessages(prev => [...prev, { id: createMessageId(), role: 'system', content: `Error resetting memory: ${getErrorMessage(error)}` }]);
      }
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <FrameworkHeader 
        title="Custom Chat UI (Vanilla React)"
        color="orange"
        settings={settings}
        onSettingsClick={() => setShowSettings(!showSettings)}
        apiStatus={apiStatus}
        apiCapabilities={apiCapabilities}
      />
      
      <div className="flex-1 overflow-hidden p-4">
        <div className="h-full bg-gradient-to-b from-orange-50 to-white rounded-lg border overflow-hidden flex flex-col">
          <div className="flex-1 overflow-y-auto p-4">
            {messages?.map((message) => (
              <div key={message.id} className={`mb-4 ${message.role === 'user' ? 'text-right' : message.role === 'system' ? 'text-center' : 'text-left'}`}>
                <div className={`inline-block px-4 py-2 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-orange-600 text-white max-w-md'
                    : message.role === 'system'
                    ? 'bg-amber-50 text-amber-900 border border-amber-200 max-w-3xl text-left whitespace-pre-wrap'
                    : 'bg-orange-100 text-gray-900 max-w-md whitespace-pre-wrap'
                }`}>
                  {message.role === 'assistant' && (
                    <div className="text-xs font-semibold mb-1 text-orange-700">Custom Assistant</div>
                  )}
                  {message.role === 'system' && (
                    <div className="text-xs font-semibold uppercase tracking-wide text-amber-700 mb-2">Conversation History</div>
                  )}
                  {message.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="text-left mb-4">
                <div className="inline-block bg-orange-100 px-4 py-2 rounded-lg">
                  <div className="text-xs font-semibold mb-1 text-orange-700">Custom Assistant</div>
                  <div className="animate-pulse">Processing your request...</div>
                </div>
              </div>
            )}
          </div>
          
          <div className="border-t bg-white p-4">
            <form onSubmit={handleSubmit}>
              <div className="flex space-x-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={`Ask questions... ${apiCapabilities.hasMemory ? `(Session: ${settings.sessionId})` : ''}`}
                  disabled={isLoading}
                  className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-1 focus:ring-orange-500"
                />
                <button
                  type="submit"
                  disabled={isLoading || !input?.trim()}
                  className="bg-orange-600 hover:bg-orange-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg"
                >
                  Send
                </button>
              </div>
            </form>
            <div className="text-xs text-gray-500 mt-2">
              Custom React chat interface • No external UI framework
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

// ============================================================================
// MAIN APP
// ============================================================================

const App = () => {
  const [activeFramework, setActiveFramework] = useState('langchain');

  const frameworks = [
    { id: 'langchain', label: 'LangChain Agent UI', color: 'blue' },
    { id: 'llamaindex', label: 'LlamaIndex Chat UI', color: 'purple' },
    { id: 'assistant', label: 'Assistant UI', color: 'green' },
    { id: 'custom', label: 'Custom Chat UI', color: 'orange' }    
  ];

  return (
    <div className="h-screen">
      <div className="bg-gray-900 text-white p-4">
        <div className="flex space-x-4 flex-wrap">
          {frameworks.map(framework => (
            <button
              key={framework.id}
              onClick={() => setActiveFramework(framework.id)}
              className={`px-4 py-2 rounded transition-colors ${
                activeFramework === framework.id 
                  ? `bg-${framework.color}-600` 
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              {framework.label}
            </button>
          ))}
        </div>
      </div>

      {activeFramework === 'langchain' && <LangChainPage />}
      {activeFramework === 'llamaindex' && <LlamaIndexPage />}
      {activeFramework === 'assistant' && <AssistantUIPage />}
      {activeFramework === 'custom' && <CustomChatPage />}      
    </div>
  );
};

export default App;
