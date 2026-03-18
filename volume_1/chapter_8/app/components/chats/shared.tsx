'use client';

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Settings } from 'lucide-react';
import {
  buildDetails,
  extractApiErrorMessage,
  extractDetailsFromContent,
  getDisplayContentFromStructuredText,
  mergeDetails,
  parseJsonText,
  processApiResponse,
  readResponsePayload,
} from '../../lib/response-normalizer';

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
  details?: ResponseDetails;
};

type TokenUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

type ToolCallDetails = {
  name: string;
  arguments?: Record<string, any>;
  output?: any;
};

type CoagentCallDetails = Record<string, any>;

type ResponseDetails = {
  provider?: string;
  sessionId?: string;
  summary?: string;
  distilled?: string;
  keywords?: string[];
  confidence?: string;
  notes?: string;
  tokenUsage?: TokenUsage;
  tokensWithMemoryRetrieval?: number;
  tokensWithoutMemoryRetrieval?: number;
  estimatedTokensSaved?: number;
  estimatedTokenReductionPercent?: number;
  wikipediaSummary?: string;
  wikipediaUrl?: string;
  wikipediaImages?: string[];
  toolCalls?: ToolCallDetails[];
  coagentCalls?: CoagentCallDetails[];
};

type ProcessedResponse = {
  content: string;
  details?: ResponseDetails;
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
  hasCoagent: boolean;
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
const LANGGRAPH_ASSISTANT_ID = 'chat';

// ============================================================================
// SHARED HOOKS AND UTILITIES
// ============================================================================

// Response processing utilities are shared across standard and streaming paths.

const parseMessageEnvelope = (text: string): ProcessedResponse | null => {
  const parsed = parseJsonText(text);
  if (!parsed || typeof parsed !== 'object') return null;
  if (typeof parsed.content !== 'string') return null;

  return {
    content: parsed.content,
    details: parsed.details
  };
};


const createMessageId = () => Date.now() + Math.floor(Math.random() * 100000);

const ThinkingIndicator = ({ label = 'Thinking...' }: { label?: string }) => (
  <span className="inline-flex items-center gap-2 text-sm text-gray-500" aria-live="polite">
    <span className="h-2 w-2 animate-pulse rounded-full bg-current" />
    <span className="animate-pulse">{label}</span>
  </span>
);

const getTurnText = (content: unknown): string => {
  if (typeof content === 'string') return content;

  if (Array.isArray(content)) {
    return content
      .map((item: any) => {
        if (typeof item === 'string') return item;
        if (item?.type === 'text' && typeof item?.text === 'string') return item.text;
        if (typeof item?.content === 'string') return item.content;
        return '';
      })
      .filter(Boolean)
      .join('\n');
  }

  if (content && typeof content === 'object' && 'text' in (content as any) && typeof (content as any).text === 'string') {
    return (content as any).text;
  }

  return '';
};

const historyMetadataLines = (turn: any): string[] => {
  const lines: string[] = [];
  const text = getTurnText(turn?.content);
  const structured = parseJsonText(text);
  const details = buildDetails(structured, turn);

  if (details?.distilled) lines.push(`   Distilled: ${details.distilled}`);
  if (details?.summary) lines.push(`   Summary: ${details.summary}`);
  if (details?.keywords?.length) lines.push(`   Keywords: ${details.keywords.join(', ')}`);
  if (details?.confidence) lines.push(`   Confidence: ${details.confidence}`);
  if (details?.notes) lines.push(`   Notes: ${details.notes}`);

  if (details?.tokenUsage?.total_tokens !== undefined) {
    lines.push(`   Tokens: total ${details.tokenUsage.total_tokens}`);
  }
  if (details?.tokenUsage?.prompt_tokens !== undefined) {
    lines.push(`   Prompt tokens: ${details.tokenUsage.prompt_tokens}`);
  }
  if (details?.tokenUsage?.completion_tokens !== undefined) {
    lines.push(`   Completion tokens: ${details.tokenUsage.completion_tokens}`);
  }

  return lines;
};

const formatHistoryMessage = (
  provider: string,
  sessionId: string,
  turns: Array<{ role: string; content?: unknown; token_usage?: TokenUsage }> = []
) => {
  if (!turns.length) {
    return `History for ${provider} (session: ${sessionId})\n\nNo conversation history found.`;
  }

  const lines = turns.map((turn, index) => {
    const roleLabel = turn.role === 'assistant' || turn.role === 'ai'
      ? 'Assistant'
      : turn.role === 'user' || turn.role === 'human'
      ? 'User'
      : turn.role;

    const content = getTurnText(turn.content) || '(no content)';
    const metadata = roleLabel === 'Assistant' ? historyMetadataLines(turn) : [];

    return [`${index + 1}. ${roleLabel}: ${content}`, ...metadata].join('\n');
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
    hasStreaming: false,
    hasCoagent: false
  });
  const [settings, setSettings] = useState<APISettings>({
    queryMode: 'single',
    selectedProvider: 'openai',
    temperature: 0.7,
    maxTokens: 1000,
    sessionId: 'default', // Added session ID support
    responseMode: 'stream' // stream | standard
  });
  const failedHealthChecksRef = useRef(0);

  const markApiHealthy = () => {
    failedHealthChecksRef.current = 0;
    setApiStatus('online');
  };

  const markApiCheckFailure = () => {
    failedHealthChecksRef.current += 1;
    if (failedHealthChecksRef.current >= 3) {
      setApiStatus('offline');
      return;
    }

    setApiStatus((prev) => (prev === 'online' ? 'online' : 'checking'));
  };

  const checkApiStatus = async () => {
    try {
      const statusResponse = await fetch(`${GATEWAY_API_BASE}/`, {
        method: 'GET',
        signal: AbortSignal.timeout(8000)
      });

      if (!statusResponse.ok) {
        markApiCheckFailure();
        return;
      }

      const statusData = await statusResponse.json();

      let providersData: { providers?: Provider[]; framework?: string } | null = null;
      try {
        const providersResponse = await fetch(`${GATEWAY_API_BASE}/providers`, {
          method: 'GET',
          signal: AbortSignal.timeout(8000)
        });

        if (providersResponse.ok) {
          providersData = await providersResponse.json();
        }
      } catch (_) {
        providersData = null;
      }

      if (providersData?.providers?.length) {
        setProviders(providersData.providers);

        const providerNames = providersData.providers.map((provider: Provider) => provider.name);
        const defaultProvider = providerNames.includes('openai') ? 'openai' : providersData.providers[0].name;
        setSettings((prev) => {
          if (providerNames.includes(prev.selectedProvider)) {
            return prev;
          }

          return { ...prev, selectedProvider: defaultProvider };
        });
      }

      let capabilitiesData = null;
      try {
        const capabilitiesResponse = await fetch(`${GATEWAY_API_BASE}/capabilities`, {
          method: 'GET',
          signal: AbortSignal.timeout(5000)
        });
        if (capabilitiesResponse.ok) {
          capabilitiesData = await capabilitiesResponse.json();
        }
      } catch (_) {
        capabilitiesData = null;
      }

      const memoryEnabled = Boolean(
        statusData.framework?.includes('History')
        || statusData?.memory
        || statusData?.memory_retrieval
        || capabilitiesData?.memory
        || capabilitiesData?.memory_retrieval
      );

      setApiCapabilities((prev) => ({
        hasMemory: memoryEnabled,
        hasHistory: memoryEnabled,
        framework: statusData.framework || providersData?.framework || '',
        hasStreaming: capabilitiesData ? Boolean(capabilitiesData?.streaming) : prev.hasStreaming,
        hasCoagent: capabilitiesData ? Boolean(capabilitiesData?.coagent) : prev.hasCoagent
      }));

      markApiHealthy();
    } catch (_) {
      markApiCheckFailure();
      // Keep current user-selected settings sticky even if API health check fails.
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
const callAPI = async (message: string, settings: APISettings, options: CallAPIOptions = {}): Promise<ProcessedResponse> => {
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
    try {
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
      let streamDetails: ResponseDetails | undefined;

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
            let parsed: any;
            try {
              parsed = JSON.parse(payloadLine);
            } catch {
              // Ignore non-JSON SSE heartbeat/done payloads.
              continue;
            }

            if (parsed.type === 'chunk') {
              fullText += parsed.content || '';
              options.onChunk?.(fullText);
            } else if (parsed.type === 'done') {
              streamDetails = mergeDetails(buildDetails(parsed.response, parsed), extractDetailsFromContent(fullText));
            } else if (parsed.type === 'error') {
              throw new Error(parsed.error || 'Streaming failed');
            }
          }
        }
      }

      return { content: getDisplayContentFromStructuredText(fullText), details: mergeDetails(streamDetails, extractDetailsFromContent(fullText)) };
    } catch (streamError) {
      console.warn('Streaming request failed, retrying with standard response mode.', streamError);
    }
  }

  const response = await fetch(`${GATEWAY_API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  const { data } = await readResponsePayload(response);
  if (!response.ok) {
    throw new Error(extractApiErrorMessage(data, response.status));
  }

  return processApiResponse(data, settings.queryMode);
};


const formatToolValue = (value: unknown): string => {
  if (value === undefined) return 'Not available';
  if (typeof value === 'string') return value;

  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

const getToolCallArguments = (toolCall: ToolCallDetails): Record<string, any> | undefined => {
  if (!toolCall.arguments || typeof toolCall.arguments !== 'object') return undefined;
  return toolCall.arguments;
};

const getWikipediaToolOutput = (toolCall: ToolCallDetails): Record<string, any> | undefined => {
  if (toolCall.name !== 'get_wikipedia_evidence_pack') return undefined;

  if (toolCall.output && typeof toolCall.output === 'object') {
    return toolCall.output as Record<string, any>;
  }

  const args = getToolCallArguments(toolCall) || {};
  const hasWikipediaPayload = 'summary' in args || 'page_url' in args || 'media' in args || 'references' in args || 'error' in args;
  return hasWikipediaPayload ? args : undefined;
};

const getWikipediaToolRequest = (toolCall: ToolCallDetails): Record<string, any> | undefined => {
  const args = getToolCallArguments(toolCall) || {};
  const wikipediaOutput = getWikipediaToolOutput(toolCall);

  if (!wikipediaOutput) {
    return Object.keys(args).length ? args : undefined;
  }

  const requestEntries = Object.entries(args).filter(([key]) => !['summary', 'page_url', 'references', 'media', 'error', 'topic'].includes(key));
  if (!requestEntries.length) return undefined;
  return Object.fromEntries(requestEntries);
};

const ToolRequestSummary = ({ request }: { request?: Record<string, any> }) => {
  if (!request || Object.keys(request).length === 0) return null;

  return (
    <div className="mt-3 flex flex-wrap gap-2 text-xs text-gray-600">
      {Object.entries(request).map(([key, value]) => (
        <div key={key} className="rounded-full bg-gray-100 px-2.5 py-1">
          <span className="font-medium text-gray-700">{key}:</span> {typeof value === 'string' ? value : formatToolValue(value)}
        </div>
      ))}
    </div>
  );
};

const toToolOutputObject = (toolCall: ToolCallDetails): Record<string, any> | undefined => {
  if (toolCall.output && typeof toolCall.output === 'object' && !Array.isArray(toolCall.output)) {
    return toolCall.output as Record<string, any>;
  }

  const args = getToolCallArguments(toolCall);
  if (!args) return undefined;

  const resemblesOutput = ['summary', 'page_url', 'media', 'references', 'images', 'url', 'source_url', 'error', 'content', 'text', 'description']
    .some((key) => key in args);

  return resemblesOutput ? args : undefined;
};

const collectImageUrls = (value: unknown): string[] => {
  const urls: string[] = [];
  const addUrl = (candidate: unknown) => {
    if (typeof candidate !== 'string') return;
    const trimmed = candidate.trim();
    if (!trimmed.startsWith('http') || urls.includes(trimmed)) return;
    urls.push(trimmed);
  };

  if (Array.isArray(value)) {
    value.forEach((item) => {
      if (typeof item === 'string') {
        addUrl(item);
      } else if (item && typeof item === 'object') {
        addUrl((item as Record<string, unknown>).url);
        addUrl((item as Record<string, unknown>).src);
      }
    });
  } else if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    collectImageUrls(record.images).forEach(addUrl);
    collectImageUrls(record.commons_images).forEach(addUrl);
    addUrl(record.url);
    addUrl(record.src);
  }

  return urls.slice(0, 8);
};

const pickFirstString = (...values: unknown[]): string | undefined => {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) return value.trim();
  }
  return undefined;
};

const prettifyToolName = (name: string): string => name
  .replace(/^get_/, '')
  .replace(/_/g, ' ')
  .replace(/\b\w/g, (char) => char.toUpperCase());

const getToolCardData = (toolCall: ToolCallDetails, index: number) => {
  const output = toolCall.name === 'get_wikipedia_evidence_pack'
    ? getWikipediaToolOutput(toolCall)
    : toToolOutputObject(toolCall);
  const request = toolCall.name === 'get_wikipedia_evidence_pack'
    ? getWikipediaToolRequest(toolCall)
    : getToolCallArguments(toolCall);
  const title = toolCall.name === 'get_wikipedia_evidence_pack'
    ? 'Wikipedia evidence'
    : pickFirstString(output?.title, output?.name, output?.topic, output?.source_title, `${prettifyToolName(toolCall.name)} result ${index + 1}`) as string;
  const text = pickFirstString(
    output?.summary,
    output?.text,
    output?.content,
    output?.description,
    output?.extract,
    output?.snippet,
    output?.answer,
    output?.error,
    typeof toolCall.output === 'string' ? toolCall.output : undefined,
  ) || (toolCall.output !== undefined ? formatToolValue(toolCall.output) : 'Awaiting tool output.');
  const sourceUrl = pickFirstString(output?.page_url, output?.source_url, output?.url, output?.link, output?.reference_url);
  const images = collectImageUrls(output?.media || output?.images || output);

  return { title, text, sourceUrl, images, request, output, rawOutput: toolCall.output };
};

const StandardizedToolCard = ({ toolCall, index, compact = false }: { toolCall: ToolCallDetails; index: number; compact?: boolean }) => {
  const { title, text, sourceUrl, images, request, rawOutput } = getToolCardData(toolCall, index);

  return (
    <section className={`rounded-xl border border-sky-100 bg-sky-50 ${compact ? 'p-3 text-xs' : 'p-4 text-sm'} text-sky-900`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="font-semibold">{title}</div>
          <div className="mt-1 text-[11px] uppercase tracking-wide text-sky-700">{prettifyToolName(toolCall.name)}</div>
        </div>
        <code className="rounded-full bg-white/70 px-2 py-1 text-[11px] text-sky-800">#{index + 1}</code>
      </div>

      <ToolRequestSummary request={request} />

      {text && <div className={`mt-2 whitespace-pre-wrap ${compact ? 'line-clamp-5' : ''} text-sky-950`}>{text}</div>}

      {sourceUrl && (
        <a
          href={sourceUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-2 inline-block text-sky-700 underline"
        >
          Open source page
        </a>
      )}

      {images.length > 0 && (
        <div className="mt-3 flex gap-2 overflow-x-auto pb-1">
          {images.map((imageUrl, imageIndex) => (
            <a key={`${imageUrl}-${imageIndex}`} href={imageUrl} target="_blank" rel="noopener noreferrer" className="shrink-0">
              <img
                src={imageUrl}
                alt={`${title} media ${imageIndex + 1}`}
                className="h-16 w-24 rounded border border-sky-200 object-cover"
                loading="lazy"
              />
            </a>
          ))}
        </div>
      )}

      {!compact && rawOutput !== undefined && (
        <details className="mt-3">
          <summary className="cursor-pointer text-xs text-sky-700 hover:text-sky-900">View raw tool payload</summary>
          <pre className="mt-2 max-h-80 overflow-auto rounded-lg bg-slate-950 p-3 text-[11px] text-slate-100">{formatToolValue(rawOutput)}</pre>
        </details>
      )}
    </section>
  );
};

const CoagentCallCard = ({ coagentCall, index }: { coagentCall: CoagentCallDetails; index: number }) => (
  <section className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
    <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Co-agent step {index + 1}</div>
    <pre className="mt-2 max-h-80 overflow-auto rounded-lg bg-slate-950 p-3 text-[11px] text-slate-100">{formatToolValue(coagentCall)}</pre>
  </section>
);

const CoagentActivitySidebar = ({ coagentCalls }: { coagentCalls?: CoagentCallDetails[] }) => {
  const items = coagentCalls?.length ? coagentCalls : [];

  return (
    <aside className="h-full min-h-0 rounded-xl border border-gray-200 bg-gray-50/80 backdrop-blur">
      <div className="border-b border-gray-200 px-4 py-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Cogent</div>
        <div className="mt-1 text-sm font-semibold text-gray-900">Co-agent</div>
        <div className="text-xs text-gray-500">This co-agent lets you inspect an agent workflow and guide its course as a human in the loop.</div>
      </div>

      <div className="h-[calc(100%-76px)] overflow-y-auto p-4">
        {items.length > 0 ? (
          <div className="space-y-4">
            {items.map((coagentCall, index) => (
              <CoagentCallCard key={`coagent-${index}`} coagentCall={coagentCall} index={index} />
            ))}
          </div>
        ) : null}
      </div>
    </aside>
  );
};

const ResponseDetailsPanel = ({ details }: { details?: ResponseDetails }) => {
  if (!details) return null;

  const tokenText = details.tokenUsage?.total_tokens !== undefined
    ? `${details.tokenUsage.total_tokens} tokens`
    : null;
  const keywords = details.keywords?.length ? details.keywords : [];
  const tokenSavingsText = details.estimatedTokenReductionPercent !== undefined
    ? `Tokens saved: ${details.estimatedTokenReductionPercent}%`
    : null;
  const toolCalls = details.toolCalls?.length ? details.toolCalls : [];

  return (
    <div className="mt-2 border-t border-gray-200 pt-2 text-xs text-gray-600">
      <div className="mb-2 flex items-end justify-between gap-2">
        <div className="flex flex-wrap gap-1">
          {keywords.map((keyword) => (
            <button
              key={keyword}
              type="button"
              className="rounded-full border border-gray-300 bg-white px-2 py-0.5 text-xs text-gray-700"
            >
              {keyword}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2">
          {tokenText && <span className="font-medium text-gray-500">{tokenText}</span>}
          {tokenSavingsText && <span className="font-medium text-gray-500">{tokenSavingsText}</span>}
        </div>
      </div>

      {toolCalls.length > 0 && (
        <div className="mb-2 space-y-2">
          {toolCalls.map((toolCall, index) => (
            <StandardizedToolCard key={`${toolCall.name}-${index}`} toolCall={toolCall} index={index} compact />
          ))}
        </div>
      )}

      {(details.summary || details.distilled || details.notes || details.confidence || details.provider || details.sessionId || details.tokensWithMemoryRetrieval !== undefined || details.tokensWithoutMemoryRetrieval !== undefined || details.estimatedTokensSaved !== undefined || details.estimatedTokenReductionPercent !== undefined) && (
        <details className="mt-2">
          <summary className="cursor-pointer text-gray-500 hover:text-gray-700">More response details</summary>
          <div className="mt-1 space-y-1 text-gray-700">
            {details.distilled && <div><span className="font-semibold">Distilled:</span> {details.distilled}</div>}
            {details.summary && <div><span className="font-semibold">Summary:</span> {details.summary}</div>}
            {details.confidence && <div><span className="font-semibold">Confidence:</span> {details.confidence}</div>}
            {details.notes && <div><span className="font-semibold">Notes:</span> {details.notes}</div>}
            {details.provider && <div><span className="font-semibold">Provider:</span> {details.provider}</div>}
            {details.sessionId && <div><span className="font-semibold">Session:</span> {details.sessionId}</div>}
            {details.tokenUsage?.prompt_tokens !== undefined && (
              <div><span className="font-semibold">Prompt tokens:</span> {details.tokenUsage.prompt_tokens}</div>
            )}
            {details.tokenUsage?.completion_tokens !== undefined && (
              <div><span className="font-semibold">Completion tokens:</span> {details.tokenUsage.completion_tokens}</div>
            )}
            {details.tokensWithMemoryRetrieval !== undefined && (
              <div><span className="font-semibold">Tokens with memory retrieval:</span> {details.tokensWithMemoryRetrieval}</div>
            )}
            {details.tokensWithoutMemoryRetrieval !== undefined && (
              <div><span className="font-semibold">Tokens without memory retrieval:</span> {details.tokensWithoutMemoryRetrieval}</div>
            )}
            {details.estimatedTokensSaved !== undefined && (
              <div><span className="font-semibold">Estimated tokens saved:</span> {details.estimatedTokensSaved}</div>
            )}
            {details.estimatedTokenReductionPercent !== undefined && (
              <div><span className="font-semibold">Estimated token reduction savings :</span> {details.estimatedTokenReductionPercent}%</div>
            )}
          </div>
        </details>
      )}
    </div>
  );
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
}) => {
  const handleHistoryButtonClick = (action: 'show' | 'reset') => {
    onClose();
    void onHistoryAction(action, settings.selectedProvider, settings.sessionId);
  };

  return (
    <div
      className={`fixed right-0 top-0 h-full w-80 bg-white border-l shadow-xl transform transition-transform z-50 ${
        isOpen ? 'translate-x-0' : 'translate-x-full'
      }`}
    >
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
                onClick={() => handleHistoryButtonClick('show')}
                className="group flex flex-1 items-center justify-between px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1"
                disabled={apiStatus !== 'online' || !settings.selectedProvider}
              >
                <span>Show History</span>
                <span aria-hidden="true" className="transition-transform group-hover:translate-x-0.5 group-focus-visible:translate-x-0.5">→</span>
              </button>
              <button
                onClick={() => handleHistoryButtonClick('reset')}
                className="group flex flex-1 items-center justify-between px-3 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-1"
                disabled={apiStatus !== 'online' || !settings.selectedProvider}
              >
                <span>Reset Memory</span>
                <span aria-hidden="true" className="transition-transform group-hover:translate-x-0.5 group-focus-visible:translate-x-0.5">→</span>
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
};

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

export {
  GATEWAY_API_BASE,
  LANGGRAPH_API_URL,
  LANGGRAPH_ASSISTANT_ID,
  parseMessageEnvelope,
  extractDetailsFromContent,
  createMessageId,
  ThinkingIndicator,
  formatHistoryMessage,
  useAPISettings,
  callAPI,
  ResponseDetailsPanel,
  CoagentActivitySidebar,
  getHistory,
  resetMemory,
  SettingsSidebar,
  FrameworkHeader,
  getErrorMessage
};

export type {
  ChatRole,
  QueryMode,
  ResponseMode,
  APIStatus,
  Provider,
  ChatMessage,
  TokenUsage,
  ResponseDetails,
  ToolCallDetails,
  CoagentCallDetails,
  ProcessedResponse,
  APISettings,
  APICapabilities
};
