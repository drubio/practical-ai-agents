export type TokenUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

export type ToolCallDetails = {
  name: string;
  arguments?: Record<string, any>;
  output?: any;
};

export type CoagentCallDetails = Record<string, any>;

export type ResponseDetails = {
  provider?: string;
  model?: string;
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

export type ProcessedResponse = {
  content: string;
  details?: ResponseDetails;
};

const normalizeToObject = (value: any): Record<string, any> | undefined => {
  if (!value) return undefined;
  if (typeof value === 'object') return value;
  if (typeof value !== 'string') return undefined;

  const parsed = parseJsonText(value);
  return parsed && typeof parsed === 'object' ? parsed : undefined;
};

const extractKeywords = (...sources: any[]): string[] | undefined => {
  const candidate = sources.find((source) => Array.isArray(source));
  if (!candidate) return undefined;

  const normalized = candidate
    .map((keyword: unknown) => (typeof keyword === 'string' ? keyword.trim() : String(keyword ?? '').trim()))
    .filter(Boolean);

  return normalized.length ? normalized : undefined;
};


const extractMediaUrls = (media: any): string[] | undefined => {
  if (!media || typeof media !== 'object') return undefined;

  const urls: string[] = [];
  const addUrl = (candidate: unknown) => {
    if (typeof candidate !== 'string') return;
    const trimmed = candidate.trim();
    if (!trimmed.startsWith('http')) return;
    if (!urls.includes(trimmed)) urls.push(trimmed);
  };

  for (const image of media?.images || []) addUrl(image);
  for (const item of media?.commons_images || []) addUrl(item?.url);

  return urls.length ? urls : undefined;
};


const normalizeToolCalls = (...sources: any[]): ToolCallDetails[] | undefined => {
  const listCandidate = sources.find((source) => Array.isArray(source));
  if (!Array.isArray(listCandidate)) return undefined;

  const normalized = listCandidate
    .filter((item) => item && typeof item === 'object' && item.name)
    .map((item) => ({
      name: String(item.name),
      arguments: item.arguments && typeof item.arguments === 'object' ? item.arguments : undefined,
      output: typeof item.output === 'string' ? (parseJsonText(item.output) ?? item.output) : item.output
    }));

  return normalized.length ? normalized : undefined;
};

const normalizeCoagentCalls = (...sources: any[]): CoagentCallDetails[] | undefined => {
  const listCandidate = sources.find((source) => Array.isArray(source));
  if (!Array.isArray(listCandidate)) return undefined;

  const normalized = listCandidate
    .filter((item) => item && typeof item === 'object')
    .map((item) => ({ ...item }));

  return normalized.length ? normalized : undefined;
};

const extractWikipediaToolOutput = (toolCalls?: ToolCallDetails[]): Record<string, any> | undefined => {
  if (!toolCalls?.length) return undefined;
  const wikipediaToolCall = toolCalls.find((toolCall) => toolCall.name === 'get_wikipedia_evidence_pack');
  return normalizeToObject(wikipediaToolCall?.output);
};

const shouldKeepNote = (note?: string): boolean => {
  if (!note) return false;
  const trimmed = note.trim();
  if (!trimmed) return false;
  return !trimmed.startsWith('Failed to parse structured JSON response');
};

const parseLabeledStructuredText = (value: string): any | null => {
  const trimmed = value.trim();
  if (!trimmed) return null;

  const fields: Record<string, any> = {};
  const extractSection = (label: string): string | undefined => {
    const pattern = new RegExp(`(?:^|\\n)${label}:\\s*([\\s\\S]*?)(?=\\n(?:Distilled|Summary|Keywords|Confidence|Notes):|$)`, 'i');
    const match = trimmed.match(pattern);
    return match?.[1]?.trim();
  };

  const distilled = extractSection('Distilled');
  if (distilled) fields.distilled = distilled;

  const summary = extractSection('Summary');
  if (summary) fields.summary = summary;

  const confidence = extractSection('Confidence');
  const notes = extractSection('Notes');
  const keywordsRaw = extractSection('Keywords');

  if (keywordsRaw) {
    const cleaned = keywordsRaw
      .replace(/^\[|\]$/g, '')
      .split(/,|\n|•|-/)
      .map((item) => item.trim())
      .filter(Boolean);

    if (cleaned.length) fields.keywords = cleaned;
  }

  const metadata: Record<string, string> = {};
  if (confidence) metadata.confidence = confidence;
  if (shouldKeepNote(notes)) metadata.notes = notes as string;

  if (Object.keys(metadata).length) {
    fields.metadata = metadata;
  }

  if (!fields.distilled && !fields.summary && !fields.keywords && !fields.metadata) {
    return null;
  }

  fields.answer = fields.summary || fields.distilled || trimmed;
  return fields;
};

export const parseJsonText = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) return null;

  try {
    return JSON.parse(trimmed);
  } catch (_) {
    const fenced = trimmed.match(/```json\s*([\s\S]*?)```/i);
    if (fenced?.[1]) {
      try {
        return JSON.parse(fenced[1].trim());
      } catch {
        // fallthrough
      }
    }

    return parseLabeledStructuredText(trimmed);
  }
};

export const extractTokenUsage = (data: any): TokenUsage | undefined => {
  const recentTurn = data?.turns?.findLast?.((turn: any) => turn?.token_usage || turn?.tokenUsage);
  const usage = data?.token_usage
    || data?.tokenUsage
    || data?.response?.token_usage
    || data?.response?.tokenUsage
    || recentTurn?.token_usage
    || recentTurn?.tokenUsage;

  if (!usage || typeof usage !== 'object') return undefined;
  return {
    prompt_tokens: usage.prompt_tokens,
    completion_tokens: usage.completion_tokens,
    total_tokens: usage.total_tokens
  };
};

export const buildDetails = (structured: any, data: any): ResponseDetails | undefined => {
  if (!structured && !data) return undefined;

  const structuredData = normalizeToObject(structured)
    || normalizeToObject(data?.response)
    || normalizeToObject(data?.content)
    || normalizeToObject(data?.answer)
    || {};
  const responseData = normalizeToObject(data?.response) || {};
  const contentData = normalizeToObject(data?.content) || {};
  const answerData = normalizeToObject(data?.answer) || {};
  const retrievalData = normalizeToObject(
    structuredData?.metadata?.retrieval
    || responseData?.metadata?.retrieval
    || contentData?.metadata?.retrieval
    || answerData?.metadata?.retrieval
    || data?.metadata?.retrieval
    || data?.response?.metadata?.retrieval
    || data?.retrieval
  ) || {};

  const note = structuredData?.metadata?.notes;
  const toolCalls = normalizeToolCalls(
    structuredData?.tool_calls,
    responseData?.tool_calls,
    contentData?.tool_calls,
    answerData?.tool_calls,
    data?.tool_calls,
    data?.response?.tool_calls
  );
  const toolOutput = extractWikipediaToolOutput(toolCalls) || {};
  const coagentCalls = normalizeCoagentCalls(
    structuredData?.coagent_calls,
    structuredData?.coagentCalls,
    responseData?.coagent_calls,
    responseData?.coagentCalls,
    contentData?.coagent_calls,
    contentData?.coagentCalls,
    answerData?.coagent_calls,
    answerData?.coagentCalls,
    data?.coagent_calls,
    data?.coagentCalls,
    data?.response?.coagent_calls,
    data?.response?.coagentCalls
  );

  const details: ResponseDetails = {
    provider: data?.provider,
    model: data?.model,
    sessionId: data?.session_id || data?.sessionId,
    summary: structuredData?.summary,
    distilled: structuredData?.distilled,
    keywords: extractKeywords(
      structuredData?.keywords,
      structuredData?.metadata?.keywords,
      structuredData?.answer?.keywords,
      responseData?.keywords,
      responseData?.metadata?.keywords,
      contentData?.keywords,
      contentData?.metadata?.keywords,
      answerData?.keywords,
      data?.keywords,
      data?.metadata?.keywords,
      data?.response?.keywords,
      data?.response?.metadata?.keywords
    ),
    confidence: structuredData?.metadata?.confidence,
    notes: shouldKeepNote(note) ? note : undefined,
    tokenUsage: extractTokenUsage(data),
    tokensWithMemoryRetrieval: retrievalData?.tokens_with_memory_retrieval,
    tokensWithoutMemoryRetrieval: retrievalData?.tokens_without_memory_retrieval,
    estimatedTokensSaved: retrievalData?.estimated_tokens_saved,
    estimatedTokenReductionPercent: retrievalData?.estimated_token_reduction_percent,
    wikipediaSummary: toolOutput?.summary,
    wikipediaUrl: toolOutput?.page_url,
    wikipediaImages: extractMediaUrls(toolOutput?.media),
    toolCalls,
    coagentCalls
  };

  return Object.values(details).some((value) => value !== undefined) ? details : undefined;
};

export const mergeDetails = (base?: ResponseDetails, fallback?: ResponseDetails): ResponseDetails | undefined => {
  if (!base && !fallback) return undefined;

  const merged: ResponseDetails = {
    provider: base?.provider || fallback?.provider,
    model: base?.model || fallback?.model,
    sessionId: base?.sessionId || fallback?.sessionId,
    summary: base?.summary || fallback?.summary,
    distilled: base?.distilled || fallback?.distilled,
    keywords: (base?.keywords && base.keywords.length ? base.keywords : fallback?.keywords),
    confidence: base?.confidence || fallback?.confidence,
    notes: base?.notes || fallback?.notes,
    tokenUsage: base?.tokenUsage || fallback?.tokenUsage,
    tokensWithMemoryRetrieval: base?.tokensWithMemoryRetrieval ?? fallback?.tokensWithMemoryRetrieval,
    tokensWithoutMemoryRetrieval: base?.tokensWithoutMemoryRetrieval ?? fallback?.tokensWithoutMemoryRetrieval,
    estimatedTokensSaved: base?.estimatedTokensSaved ?? fallback?.estimatedTokensSaved,
    estimatedTokenReductionPercent: base?.estimatedTokenReductionPercent ?? fallback?.estimatedTokenReductionPercent,
    wikipediaSummary: base?.wikipediaSummary || fallback?.wikipediaSummary,
    wikipediaUrl: base?.wikipediaUrl || fallback?.wikipediaUrl,
    wikipediaImages: (base?.wikipediaImages && base.wikipediaImages.length ? base.wikipediaImages : fallback?.wikipediaImages),
    toolCalls: (base?.toolCalls && base.toolCalls.length ? base.toolCalls : fallback?.toolCalls),
    coagentCalls: (base?.coagentCalls && base.coagentCalls.length ? base.coagentCalls : fallback?.coagentCalls)
  };

  return Object.values(merged).some((value) => value !== undefined) ? merged : undefined;
};

export const extractApiErrorMessage = (data: any, status?: number): string => {
  if (typeof data === 'string' && data.trim()) {
    return data.trim();
  }

  if (!data || typeof data !== 'object') {
    return status ? `Request failed with status ${status}` : 'Request failed';
  }

  const candidates = [data.error, data.message, data.detail, data.response?.error, data.response?.message, data.response?.detail];
  const found = candidates.find((value) => typeof value === 'string' && value.trim().length > 0);
  if (found) return found;

  if (Array.isArray(data.detail)) {
    const details = data.detail
      .map((item: any) => (typeof item === 'string' ? item : item?.msg ? String(item.msg) : ''))
      .filter(Boolean)
      .join('; ');

    if (details) return details;
  }

  return status ? `Request failed with status ${status}` : 'Request failed';
};

export const readResponsePayload = async (response: Response): Promise<{ data: any; isJson: boolean }> => {
  const rawText = await response.text();
  if (!rawText.trim()) return { data: null, isJson: false };

  try {
    return { data: JSON.parse(rawText), isJson: true };
  } catch {
    return { data: rawText, isJson: false };
  }
};

export const getDisplayContentFromStructuredText = (content: string): string => {
  const parsed = parseJsonText(content);
  if (!parsed || typeof parsed !== 'object') return content;
  if (typeof parsed.answer === 'string' && parsed.answer.trim()) return parsed.answer;
  if (typeof parsed.distilled === 'string' && parsed.distilled.trim()) return parsed.distilled;
  if (typeof parsed.summary === 'string' && parsed.summary.trim()) return parsed.summary;
  return content;
};

export const extractDetailsFromContent = (content: string): ResponseDetails | undefined => {
  const parsed = parseJsonText(content);
  if (!parsed) return undefined;
  return buildDetails(parsed, parsed);
};

const getContentFromResponseObject = (response: any, rawAnswer?: unknown, fallback = 'No response available'): string => {
  const candidates: unknown[] = [
    response?.answer,
    response?.summary,
    response?.distilled,
    response?.content,
    response?.text,
    response?.message,
    response?.final_answer,
    response?.finalAnswer,
    response?.raw_answer,
    response?.rawAnswer,
    response?.output_text,
    rawAnswer
  ];

  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim()) return candidate;
  }

  if (Array.isArray(response?.content)) {
    const joined = response.content
      .map((item: any) => {
        if (typeof item === 'string') return item;
        if (typeof item?.text === 'string') return item.text;
        if (typeof item?.content === 'string') return item.content;
        return '';
      })
      .filter(Boolean)
      .join('\n');

    if (joined.trim()) return joined;
  }


  if (response?.response && typeof response.response === 'object') {
    const nested = getContentFromResponseObject(response.response, rawAnswer, '');
    if (nested.trim()) return nested;
  }

  if (response && typeof response === 'object') {
    const structured = parseJsonText(JSON.stringify(response));
    if (typeof structured?.answer === 'string' && structured.answer.trim()) return structured.answer;
    if (typeof structured?.summary === 'string' && structured.summary.trim()) return structured.summary;
    if (typeof structured?.distilled === 'string' && structured.distilled.trim()) return structured.distilled;
  }

  return fallback;
};

export const processApiResponse = (data: any, queryMode: string): ProcessedResponse => {
  if (queryMode === 'single') {
    if (data.success) {
      if (typeof data.response === 'string') {
        const structured = parseJsonText(data.response);
        return {
          content: (structured?.answer || structured?.summary || structured?.distilled || data.response),
          details: buildDetails(structured, data)
        };
      }

      if (typeof data.response === 'object' && data.response !== null) {
        return {
          content: getContentFromResponseObject(data.response, data.raw_answer, 'No response content available'),
          details: buildDetails(data.response, data)
        };
      }

      return { content: data.raw_answer || 'No response available', details: buildDetails(undefined, data) };
    }

    return { content: `Error: ${extractApiErrorMessage(data)}` };
  }

  if (data.success) {
    let content = `Results from ${data.summary?.total_providers || Object.keys(data.responses).length} providers:\n\n`;

    for (const [provider, response] of Object.entries(data.responses)) {
      const providerResponse = response as any;
      content += `**${provider}**\n`;
      if (providerResponse.model) {
        content += `- Model: ${providerResponse.model}\n`;
      }
      content += `- Response: `;

      if (providerResponse.success) {
        if (typeof providerResponse.response === 'string') {
          content += providerResponse.response;
        } else if (typeof providerResponse.response === 'object' && providerResponse.response !== null) {
          content += getContentFromResponseObject(providerResponse.response, providerResponse.raw_answer);
        } else {
          content += providerResponse.raw_answer || 'No response available';
        }
      } else {
        content += `Error: ${providerResponse.error}`;
      }

      content += '\n\n';
    }

    return { content };
  }

  return { content: `Error: ${extractApiErrorMessage(data)}` };
};
