export type TokenUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

export type ResponseDetails = {
  provider?: string;
  sessionId?: string;
  summary?: string;
  distilled?: string;
  keywords?: string[];
  confidence?: string;
  notes?: string;
  tokenUsage?: TokenUsage;
};

export type ProcessedResponse = {
  content: string;
  details?: ResponseDetails;
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

  const note = structured?.metadata?.notes;

  const details: ResponseDetails = {
    provider: data?.provider,
    sessionId: data?.session_id || data?.sessionId,
    summary: structured?.summary,
    distilled: structured?.distilled,
    keywords: Array.isArray(structured?.keywords) ? structured.keywords.filter(Boolean) : undefined,
    confidence: structured?.metadata?.confidence,
    notes: shouldKeepNote(note) ? note : undefined,
    tokenUsage: extractTokenUsage(data)
  };

  return Object.values(details).some((value) => value !== undefined) ? details : undefined;
};

export const mergeDetails = (base?: ResponseDetails, fallback?: ResponseDetails): ResponseDetails | undefined => {
  if (!base && !fallback) return undefined;

  const merged: ResponseDetails = {
    provider: base?.provider || fallback?.provider,
    sessionId: base?.sessionId || fallback?.sessionId,
    summary: base?.summary || fallback?.summary,
    distilled: base?.distilled || fallback?.distilled,
    keywords: (base?.keywords && base.keywords.length ? base.keywords : fallback?.keywords),
    confidence: base?.confidence || fallback?.confidence,
    notes: base?.notes || fallback?.notes,
    tokenUsage: base?.tokenUsage || fallback?.tokenUsage
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
          content: data.response.answer || data.response.summary || data.raw_answer || 'No response content available',
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
      content += `**${provider}**: `;

      if (providerResponse.success) {
        if (typeof providerResponse.response === 'string') {
          content += providerResponse.response;
        } else if (typeof providerResponse.response === 'object' && providerResponse.response !== null) {
          content += providerResponse.response.answer || providerResponse.response.summary || providerResponse.raw_answer || 'No response available';
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
