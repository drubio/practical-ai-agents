/**
 * Chapter 7 tool utilities (JavaScript / Node).
 *
 * Goals:
 * - API-key-free where possible
 * - easy to wire into LangChain "tool calling" demos
 * - mirrors tools.py behavior
 *
 * Requires:
 * - Node 18+ (global fetch)
 * - Optional dependency for markdown->HTML:
 *     npm i marked
 *   If you don't install it, markdown conversion falls back to a tiny safe-ish stub.
 */

'use strict';

// Optional markdown dependency
let marked = null;
try {
  // eslint-disable-next-line import/no-extraneous-dependencies
  ({ marked } = require('marked'));
} catch (_) {
  marked = null;
}

/** @typedef {{name: string, description: string, parameters: Record<string,string>}} ToolDefinition */

/* -------------------------
 * Existing tools
 * ------------------------- */

function format_markdown_to_html(text = '') {
  const input = String(text ?? '');
  if (marked) return marked.parse(input);

  // Fallback: minimal formatting; keeps chapter runnable without extra deps.
  // (Not a full markdown parser.)
  const escaped = input
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
  // naive: paragraphs + line breaks
  return escaped
    .split(/\n{2,}/g)
    .map((p) => `<p>${p.replaceAll('\n', '<br/>')}</p>`)
    .join('\n');
}

function get_datetime(timezone = 'UTC') {
  // Using Intl time zone support (Node builds typically include ICU).
  // If timezone is invalid, Intl throws; we catch and return an error.
  try {
    const tz = String(timezone || 'UTC');
    const dtf = new Intl.DateTimeFormat('en-CA', {
      timeZone: tz,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
      timeZoneName: 'short',
    });
    // Format parts to match "YYYY-MM-DD HH:MM:SS ZZZ" similar to Python
    const parts = dtf.formatToParts(new Date());
    const byType = Object.fromEntries(parts.map((p) => [p.type, p.value]));
    const date = `${byType.year}-${byType.month}-${byType.day}`;
    const time = `${byType.hour}:${byType.minute}:${byType.second}`;
    const tzName = byType.timeZoneName || tz;
    return `${date} ${time} ${tzName}`;
  } catch (err) {
    return `Error: ${err?.message || String(err)}`;
  }
}

/* -------------------------
 * HTTP helpers (no extra deps)
 * ------------------------- */

async function http_get_json(url, timeoutMs = 10_000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: 'GET',
      headers: {
        'User-Agent': 'software-with-llms-ch7/1.0 (tool demo; https://example.invalid)',
        'Accept': 'application/json',
      },
      signal: ctrl.signal,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status} for ${url} :: ${body.slice(0, 200)}`);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

function safe_json_stringify(obj) {
  return JSON.stringify(obj, null, 2);
}

/* -------------------------
 * Wikipedia / Wikimedia tools (no API keys)
 * ------------------------- */

const WIKI_API = 'https://en.wikipedia.org/w/api.php';
const WIKI_REST = 'https://en.wikipedia.org/api/rest_v1';
const COMMONS_API = 'https://commons.wikimedia.org/w/api.php';

async function wiki_opensearch(query, limit = 1) {
  const q = encodeURIComponent(query);
  const url = `${WIKI_API}?action=opensearch&search=${q}&limit=${Number(limit)}&namespace=0&format=json&origin=*`;
  const data = await http_get_json(url);
  const titles = Array.isArray(data?.[1]) ? data[1] : [];
  const urls = Array.isArray(data?.[3]) ? data[3] : [];
  const out = [];
  for (let i = 0; i < Math.min(titles.length, urls.length); i++) {
    out.push([String(titles[i]), String(urls[i])]);
  }
  return out; // [[title, url], ...]
}

async function wiki_page_summary(title) {
  const safeTitle = encodeURIComponent(String(title).replaceAll(' ', '_'));
  const url = `${WIKI_REST}/page/summary/${safeTitle}`;
  return await http_get_json(url);
}

async function wiki_external_links(title, maxLinks = 10) {
  const t = encodeURIComponent(title);
  const url = `${WIKI_API}?action=parse&page=${t}&prop=externallinks&format=json&origin=*`;
  const data = await http_get_json(url);
  const links = data?.parse?.externallinks || [];
  const cleaned = [];
  for (const link of links) {
    if (typeof link === 'string' && link.startsWith('http')) cleaned.push(link);
    if (cleaned.length >= maxLinks) break;
  }
  return cleaned;
}

async function wiki_page_images(title, maxFiles = 8) {
  const t = encodeURIComponent(title);
  const url = `${WIKI_API}?action=query&titles=${t}&prop=images&imlimit=${Number(maxFiles)}&format=json&origin=*`;
  const data = await http_get_json(url);

  const pagesObj = data?.query?.pages || {};
  const pages = Object.values(pagesObj);

  const fileTitles = [];
  for (const page of pages) {
    const imgs = page?.images || [];
    for (const img of imgs) {
      const name = img?.title;
      if (typeof name === 'string' && name.startsWith('File:')) {
        if (/\.(png|jpg|jpeg|gif|svg|webm|ogv)$/i.test(name)) fileTitles.push(name);
      }
      if (fileTitles.length >= maxFiles) break;
    }
  }
  return await commons_resolve_file_urls(fileTitles);
}

async function commons_resolve_file_urls(fileTitles) {
  if (!Array.isArray(fileTitles) || fileTitles.length === 0) return [];
  const titles = fileTitles
    .slice(0, 20)
    .map((t) => encodeURIComponent(t))
    .join('|');

  const url = `${COMMONS_API}?action=query&titles=${titles}&prop=imageinfo&iiprop=url|mime&format=json&origin=*`;
  const data = await http_get_json(url);

  const out = [];
  const pagesObj = data?.query?.pages || {};
  const pages = Object.values(pagesObj);
  for (const page of pages) {
    const infos = page?.imageinfo || [];
    for (const ii of infos) {
      const u = ii?.url;
      if (typeof u === 'string' && u.startsWith('http')) out.push(u);
    }
  }

  // De-dupe preserving order
  const seen = new Set();
  const deduped = [];
  for (const u of out) {
    if (!seen.has(u)) {
      seen.add(u);
      deduped.push(u);
    }
  }
  return deduped;
}

async function commons_search_media(query, kind, limit = 6) {
  let q = String(query ?? '').trim();
  if (kind === 'video') q = `${q} filetype:video`;
  if (kind === 'image') q = `${q} filetype:bitmap`;

  const qEnc = encodeURIComponent(q);
  const url = `${COMMONS_API}?action=query&list=search&srsearch=${qEnc}&srlimit=${Number(limit)}&format=json&origin=*`;
  const data = await http_get_json(url);

  const results = data?.query?.search || [];
  const fileTitles = [];
  for (const r of results) {
    const title = r?.title;
    if (typeof title === 'string' && title.startsWith('File:')) {
      if (kind === 'video' && /\.(webm|ogv)$/i.test(title)) fileTitles.push(title);
      if (kind === 'image' && /\.(png|jpg|jpeg|gif|svg)$/i.test(title)) fileTitles.push(title);
    }
  }

  const urls = await commons_resolve_file_urls(fileTitles);

  const out = [];
  for (let i = 0; i < Math.min(fileTitles.length, urls.length); i++) {
    const ft = fileTitles[i];
    const u = urls[i];
    out.push({
      file_title: ft,
      file_page: `https://commons.wikimedia.org/wiki/${encodeURIComponent(ft.replaceAll(' ', '_'))}`,
      url: u,
    });
  }
  return out;
}

async function get_wikipedia_evidence_pack(query, opts = {}) {
  const q = String(query ?? '').trim();
  if (!q) return { error: 'Query is empty.' };

  const max_references = Number(opts.max_references ?? 8);
  const max_page_media = Number(opts.max_page_media ?? 6);
  const max_commons_images = Number(opts.max_commons_images ?? 6);
  const max_commons_videos = Number(opts.max_commons_videos ?? 4);

  const candidates = await wiki_opensearch(q, 1);
  if (!candidates.length) return { error: 'No Wikipedia results found.', query: q };

  const [title, pageUrl] = candidates[0];

  const summaryData = await wiki_page_summary(title);
  const summary = summaryData?.extract || '';
  const wikiUrl =
    summaryData?.content_urls?.desktop?.page ||
    summaryData?.content_urls?.mobile?.page ||
    pageUrl;

  const images = [];
  const orig = summaryData?.originalimage?.source;
  const thumb = summaryData?.thumbnail?.source;
  if (typeof orig === 'string' && orig.startsWith('http')) images.push(orig);
  if (typeof thumb === 'string' && thumb.startsWith('http') && thumb !== orig) images.push(thumb);

  const pageImages = await wiki_page_images(title, max_page_media);
  for (const u of pageImages) if (!images.includes(u)) images.push(u);

  const commonsImages = await commons_search_media(q, 'image', max_commons_images);
  const commonsVideos = await commons_search_media(q, 'video', max_commons_videos);

  const references = await wiki_external_links(title, max_references);

  return {
    query: q,
    topic: title,
    page_url: wikiUrl,
    summary,
    references,
    media: {
      images: images.slice(0, max_page_media + 2),
      commons_images: commonsImages,
      commons_videos: commonsVideos,
    },
  };
}

/* -------------------------
 * DuckDuckGo Instant Answer (no key, limited)
 * ------------------------- */

async function duckduckgo_instant_answer(query, maxRelated = 10) {
  const q = String(query ?? '').trim();
  const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(q)}&format=json&no_redirect=1&no_html=1`;
  const data = await http_get_json(url);

  const related = [];

  function pullRelated(items) {
    if (!Array.isArray(items)) return;
    for (const it of items) {
      if (it && typeof it === 'object' && Array.isArray(it.Topics)) {
        pullRelated(it.Topics);
      } else if (it && typeof it === 'object') {
        const text = it.Text;
        const firstUrl = it.FirstURL;
        if (typeof text === 'string' && typeof firstUrl === 'string') {
          related.push({ text, url: firstUrl });
        }
      }
      if (related.length >= maxRelated) return;
    }
  }

  pullRelated(data?.RelatedTopics);

  return {
    query: q,
    abstract: data?.AbstractText || '',
    abstract_url: data?.AbstractURL || '',
    heading: data?.Heading || '',
    related_topics: related.slice(0, maxRelated),
  };
}

/* -------------------------
 * Tool registry + dispatch
 * ------------------------- */

/** @type {ToolDefinition[]} */
const TOOL_DEFINITIONS = [
  {
    name: 'format_markdown_to_html',
    description: 'Convert markdown text into HTML.',
    parameters: { text: 'string - markdown content' },
  },
  {
    name: 'get_datetime',
    description: 'Get the current datetime in a timezone (e.g. UTC, Europe/Madrid).',
    parameters: { timezone: 'string - IANA timezone' },
  },
  {
    name: 'get_wikipedia_evidence_pack',
    description:
      'For ANY query: fetch Wikipedia summary + external references + related media (images and some videos via Wikimedia Commons). No API keys.',
    parameters: {
      query: 'string - any user query/topic',
      max_references: 'int - external links to return (default 8)',
      max_page_media: 'int - images sourced from page usage (default 6)',
      max_commons_images: 'int - Commons image search results (default 6)',
      max_commons_videos: 'int - Commons video search results (default 4)',
    },
  },
  {
    name: 'duckduckgo_instant_answer',
    description:
      "DuckDuckGo Instant Answer (no key). Limited: not full search results; useful for abstracts + related-topic links.",
    parameters: {
      query: 'string - any user query/topic',
      max_related: 'int - related topics (default 10)',
    },
  },
];

function list_tools() {
  return TOOL_DEFINITIONS;
}

function buildToolsPrompt() {
  const lines = [];
  for (const tool of TOOL_DEFINITIONS) {
    const params = Object.entries(tool.parameters)
      .map(([k, v]) => `${k} (${v})`)
      .join(', ');
    lines.push(`- ${tool.name}: ${tool.description} Params: ${params}`);
  }
  return lines.join('\n');
}


/**
 * Dispatch to available tools by name.
 * @param {string} action
 * @param {Record<string, any>} [parameters]
 * @returns {Promise<string>} tool output (string)
 */
async function runTool(action, parameters = {}) {
  const p = parameters || {};
  switch (action) {
    case 'format_markdown_to_html':
      return format_markdown_to_html(String(p.text ?? ''));
    case 'get_datetime':
      return get_datetime(String(p.timezone ?? 'UTC'));

    case 'get_wikipedia_evidence_pack': {
      const query = String(p.query ?? '').trim();
      const pack = await get_wikipedia_evidence_pack(query, {
        max_references: p.max_references,
        max_page_media: p.max_page_media,
        max_commons_images: p.max_commons_images,
        max_commons_videos: p.max_commons_videos,
      });
      return safe_json_stringify(pack);
    }

    case 'duckduckgo_instant_answer': {
      const query = String(p.query ?? '').trim();
      const maxRelated = Number(p.max_related ?? 10);
      const data = await duckduckgo_instant_answer(query, maxRelated);
      return safe_json_stringify(data);
    }

    default:
      return `Unknown action: ${action}`;
  }
}


export {
  list_tools,
  buildToolsPrompt,
  runTool,
  format_markdown_to_html,
  get_datetime,
  get_wikipedia_evidence_pack,
  duckduckgo_instant_answer,
};
