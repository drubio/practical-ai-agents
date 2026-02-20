/** Chapter 7 tool utilities (JavaScript / Node). */

'use strict';

/** @typedef {{name: string, description: string, parameters: Record<string,string>}} ToolDefinition */

const WIKI_API = 'https://en.wikipedia.org/w/api.php';
const WIKI_REST = 'https://en.wikipedia.org/api/rest_v1';
const COMMONS_API = 'https://commons.wikimedia.org/w/api.php';

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function wikiGetJson(baseUrl, params = {}, attempts = 3) {
  const query = new URLSearchParams(params).toString();
  const url = query ? `${baseUrl}?${query}` : baseUrl;

  let lastError = null;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10_000);

    try {
      const res = await fetch(url, {
        method: 'GET',
        headers: {
          Accept: 'application/json',
          'User-Agent': 'software-with-llms-ch7/1.0',
        },
        signal: controller.signal,
      });

      if (res.ok) {
        return await res.json();
      }

      const status = res.status;
      const responsePreview = await res.text().catch(() => '');
      const error = new Error(`HTTP ${status} for ${url} :: ${responsePreview.slice(0, 160)}`);

      if ((status === 429 || status >= 500) && attempt < attempts) {
        lastError = error;
        await sleep(250 * attempt);
        continue;
      }
      throw error;
    } catch (error) {
      lastError = error;
      if (attempt < attempts) {
        await sleep(250 * attempt);
        continue;
      }
      throw lastError;
    } finally {
      clearTimeout(timeout);
    }
  }

  throw lastError || new Error(`Failed to fetch ${url}`);
}

async function commonsResolveFileUrls(fileTitles) {
  if (!Array.isArray(fileTitles) || fileTitles.length === 0) return [];

  const data = await wikiGetJson(COMMONS_API, {
    action: 'query',
    titles: fileTitles.slice(0, 20).join('|'),
    prop: 'imageinfo',
    iiprop: 'url|mime',
    format: 'json',
    origin: '*',
  });

  const urls = [];
  for (const page of Object.values(data?.query?.pages || {})) {
    for (const info of page?.imageinfo || []) {
      const candidate = info?.url;
      if (typeof candidate === 'string' && candidate.startsWith('http') && !urls.includes(candidate)) {
        urls.push(candidate);
      }
    }
  }
  return urls;
}


function normalizeWikipediaQuery(rawQuery) {
  const input = String(rawQuery ?? '').trim();
  if (!input) return '';

  return input
    .replace(/^\s*(what\s+is|who\s+is|where\s+is|when\s+did|when\s+was|why\s+is|how\s+is)\s+/i, '')
    .replace(/[?]+$/g, '')
    .trim();
}

async function get_wikipedia_evidence_pack(query, opts = {}) {
  const q = String(query ?? '').trim();
  if (!q) return { error: 'Query is empty.' };
  const normalizedQuery = normalizeWikipediaQuery(q);

  const max_references = Number(opts.max_references ?? 8);
  const max_page_media = Number(opts.max_page_media ?? 6);
  const max_commons_images = Number(opts.max_commons_images ?? 6);
  const max_commons_videos = Number(opts.max_commons_videos ?? 4);

  let searchData = await wikiGetJson(WIKI_API, {
    action: 'opensearch',
    search: q,
    limit: 1,
    namespace: 0,
    format: 'json',
    origin: '*',
  });

  let titles = Array.isArray(searchData?.[1]) ? searchData[1] : [];
  let pageUrls = Array.isArray(searchData?.[3]) ? searchData[3] : [];

  if (!titles.length && normalizedQuery && normalizedQuery.toLowerCase() !== q.toLowerCase()) {
    searchData = await wikiGetJson(WIKI_API, {
      action: 'opensearch',
      search: normalizedQuery,
      limit: 1,
      namespace: 0,
      format: 'json',
      origin: '*',
    });
    titles = Array.isArray(searchData?.[1]) ? searchData[1] : [];
    pageUrls = Array.isArray(searchData?.[3]) ? searchData[3] : [];
  }

  if (!titles.length) {
    const fallbackSearch = await wikiGetJson(WIKI_API, {
      action: 'query',
      list: 'search',
      srsearch: normalizedQuery || q,
      srlimit: 1,
      format: 'json',
      origin: '*',
    });
    const firstResult = fallbackSearch?.query?.search?.[0];
    if (firstResult?.title) {
      titles = [String(firstResult.title)];
      pageUrls = [`https://en.wikipedia.org/wiki/${encodeURIComponent(String(firstResult.title).replaceAll(' ', '_'))}`];
    }
  }

  if (!titles.length) return { error: 'No Wikipedia results found.', query: q };

  const title = String(titles[0]);
  const fallbackPageUrl = pageUrls[0] ? String(pageUrls[0]) : '';

  const summaryData = await wikiGetJson(`${WIKI_REST}/page/summary/${encodeURIComponent(title.replaceAll(' ', '_'))}`);
  const summary = summaryData?.extract || '';
  const wikiUrl = summaryData?.content_urls?.desktop?.page || fallbackPageUrl;

  const images = [];
  for (const candidate of [summaryData?.originalimage?.source, summaryData?.thumbnail?.source]) {
    if (typeof candidate === 'string' && candidate.startsWith('http') && !images.includes(candidate)) {
      images.push(candidate);
    }
  }

  const pageImagesData = await wikiGetJson(WIKI_API, {
    action: 'query',
    titles: title,
    prop: 'images',
    imlimit: max_page_media,
    format: 'json',
    origin: '*',
  });

  const pageFileTitles = [];
  for (const page of Object.values(pageImagesData?.query?.pages || {})) {
    for (const img of page?.images || []) {
      const name = img?.title;
      if (typeof name === 'string' && name.startsWith('File:') && /\.(png|jpg|jpeg|gif|svg|webm|ogv)$/i.test(name)) {
        pageFileTitles.push(name);
      }
      if (pageFileTitles.length >= max_page_media) break;
    }
  }

  for (const url of await commonsResolveFileUrls(pageFileTitles)) {
    if (!images.includes(url)) images.push(url);
  }

  const referencesData = await wikiGetJson(WIKI_API, {
    action: 'parse',
    page: title,
    prop: 'externallinks',
    format: 'json',
    origin: '*',
  });
  const references = [];
  for (const link of referencesData?.parse?.externallinks || []) {
    if (typeof link === 'string' && link.startsWith('http')) references.push(link);
    if (references.length >= max_references) break;
  }

  async function commonsMedia(kind, limit) {
    const queryText = kind === 'video' ? `${q} filetype:video` : `${q} filetype:bitmap`;
    const commonsSearch = await wikiGetJson(COMMONS_API, {
      action: 'query',
      list: 'search',
      srsearch: queryText,
      srlimit: limit,
      format: 'json',
      origin: '*',
    });

    const fileTitles = [];
    for (const result of commonsSearch?.query?.search || []) {
      const name = result?.title;
      if (typeof name !== 'string' || !name.startsWith('File:')) continue;
      if (kind === 'video' && !/\.(webm|ogv)$/i.test(name)) continue;
      if (kind === 'image' && !/\.(png|jpg|jpeg|gif|svg)$/i.test(name)) continue;
      fileTitles.push(name);
    }

    const urls = await commonsResolveFileUrls(fileTitles);
    return fileTitles.slice(0, urls.length).map((fileTitle, idx) => ({
      file_title: fileTitle,
      file_page: `https://commons.wikimedia.org/wiki/${encodeURIComponent(fileTitle.replaceAll(' ', '_'))}`,
      url: urls[idx],
    }));
  }

  return {
    query: q,
    topic: title,
    page_url: wikiUrl,
    summary,
    references,
    media: {
      images: images.slice(0, max_page_media + 2),
      commons_images: await commonsMedia('image', max_commons_images),
      commons_videos: await commonsMedia('video', max_commons_videos),
    },
  };
}

/** @type {ToolDefinition[]} */
const TOOL_DEFINITIONS = [
  {
    name: 'get_wikipedia_evidence_pack',
    description: 'For any query: fetch Wikipedia summary + references + media from Wikimedia Commons.',
    parameters: {
      query: 'string - any user query/topic',
      max_references: 'int - external links to return (default 8)',
      max_page_media: 'int - images sourced from page usage (default 6)',
      max_commons_images: 'int - Commons image search results (default 6)',
      max_commons_videos: 'int - Commons video search results (default 4)',
    },
  },
];

function list_tools() {
  return TOOL_DEFINITIONS;
}

function buildToolsPrompt() {
  return TOOL_DEFINITIONS
    .map((tool) => {
      const params = Object.entries(tool.parameters)
        .map(([k, v]) => `${k} (${v})`)
        .join(', ');
      return `- ${tool.name}: ${tool.description} Params: ${params}`;
    })
    .join('\n');
}

async function runTool(action, parameters = {}) {
  if (action !== 'get_wikipedia_evidence_pack') {
    return `Unknown action: ${action}`;
  }

  try {
    const p = parameters || {};
    const pack = await get_wikipedia_evidence_pack(String(p.query ?? '').trim(), {
      max_references: p.max_references,
      max_page_media: p.max_page_media,
      max_commons_images: p.max_commons_images,
      max_commons_videos: p.max_commons_videos,
    });
    return JSON.stringify(pack, null, 2);
  } catch (error) {
    return JSON.stringify(
      {
        error: error?.message || String(error),
        query: String(parameters?.query ?? '').trim(),
      },
      null,
      2,
    );
  }
}

export { list_tools, buildToolsPrompt, runTool, get_wikipedia_evidence_pack };
