/**
 * Local deterministic tools for LLM agents.
 *
 * These tools are designed to integrate with LangChain or LlamaIndex agents
 * while remaining framework-neutral. They provide reliable, non-LLM utilities
 * for:
 *
 * - summarization
 * - keyword extraction
 * - task extraction
 * - priority scoring
 * - workflow routing
 * - HTML/text parsing
 * - datetime resolution
 * - JSON formatting
 * - calculations
 *
 * The Python counterpart is implemented in tools.py with equivalent names
 * using snake_case.
 */

import * as cheerio from "cheerio";
import * as chrono from "chrono-node";
import natural from "natural";

const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
  "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it",
  "its", "me", "my", "of", "on", "or", "our", "she", "so", "that", "the",
  "their", "them", "they", "this", "to", "us", "was", "we", "were", "will",
  "with", "you", "your"
]);

const TASK_PATTERNS = [
  /\bTODO\b[:\-]?\s*(.+)/i,
  /\baction item\b[:\-]?\s*(.+)/i,
  /\bneed to\b\s+(.+)/i,
  /\bmust\b\s+(.+)/i,
  /\bshould\b\s+(.+)/i,
  /\bplease\b\s+(.+)/i
];

const HIGH_PRIORITY_HINTS = [
  "urgent", "asap", "immediately", "blocker", "critical", "production",
  "outage", "security", "deadline", "today", "now", "failure", "broken"
];

const MEDIUM_PRIORITY_HINTS = [
  "soon", "review", "follow up", "follow-up", "important", "tomorrow",
  "next", "schedule", "plan", "pending"
];

function normalizeWhitespace(text) {
  return String(text ?? "").replace(/\s+/g, " ").trim();
}

function splitSentences(text) {
  const cleaned = normalizeWhitespace(text);
  if (!cleaned) {
    return [];
  }

  return cleaned
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function safeParseJsonOrString(value) {
  if (typeof value !== "string") {
    return value;
  }

  const stripped = value.trim();
  if (!stripped) {
    return value;
  }

  try {
    return JSON.parse(stripped);
  } catch {
    return value;
  }
}

export function summarizeText(text, maxSentences = 3) {
  const cleaned = normalizeWhitespace(text);
  if (!cleaned) {
    return { summary: "", sentenceCount: 0 };
  }

  const sentences = splitSentences(cleaned);
  if (sentences.length === 0) {
    return { summary: cleaned, sentenceCount: 1 };
  }

  if (sentences.length <= maxSentences) {
    return {
      summary: sentences.join(" "),
      sentenceCount: sentences.length
    };
  }

  const tfidf = new natural.TfIdf();
  sentences.forEach((sentence) => tfidf.addDocument(sentence));

  const scores = sentences.map((sentence, index) => {
    let total = 0;
    const tokens = sentence
      .toLowerCase()
      .match(/\b[a-z][a-z0-9\-]{2,}\b/g) || [];

    const uniqueTokens = [...new Set(tokens)].filter((token) => !STOPWORDS.has(token));
    uniqueTokens.forEach((token) => {
      total += tfidf.tfidf(token, index);
    });

    return { index, score: total };
  });

  const selected = scores
    .sort((a, b) => b.score - a.score)
    .slice(0, maxSentences)
    .sort((a, b) => a.index - b.index)
    .map((item) => sentences[item.index]);

  return {
    summary: selected.join(" "),
    sentenceCount: sentences.length
  };
}

export function extractKeywords(text, topK = 8) {
  const cleaned = normalizeWhitespace(text);
  if (!cleaned) {
    return { keywords: [] };
  }

  const tfidf = new natural.TfIdf();
  tfidf.addDocument(cleaned);

  const terms = tfidf
    .listTerms(0)
    .filter((item) => item.term && !STOPWORDS.has(item.term))
    .slice(0, topK)
    .map((item) => item.term);

  return { keywords: terms };
}

export function extractTasks(text) {
  const cleaned = normalizeWhitespace(text);
  if (!cleaned) {
    return { tasks: [] };
  }

  const sentences = splitSentences(cleaned);
  const tasks = [];

  for (const sentence of sentences) {
    const lowered = sentence.toLowerCase();
    let matched = false;

    for (const pattern of TASK_PATTERNS) {
      const match = sentence.match(pattern);
      if (match && match[1]) {
        tasks.push({
          task: normalizeWhitespace(match[1].replace(/\.$/, "")),
          source: sentence
        });
        matched = true;
        break;
      }
    }

    if (matched) {
      continue;
    }

    if (
      ["fix", "implement", "review", "update", "send", "draft", "prepare", "schedule"]
        .some((token) => lowered.includes(token))
    ) {
      tasks.push({
        task: sentence.replace(/\.$/, ""),
        source: sentence
      });
    }
  }

  const seen = new Set();
  const deduped = tasks.filter((item) => {
    const key = item.task.trim().toLowerCase();
    if (!key || seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });

  return { tasks: deduped };
}

export function scorePriority(text) {
  const cleaned = normalizeWhitespace(text).toLowerCase();
  if (!cleaned) {
    return { priority: "low", score: 0, reasons: [] };
  }

  let score = 0;
  const reasons = [];

  for (const hint of HIGH_PRIORITY_HINTS) {
    if (cleaned.includes(hint)) {
      score += 3;
      reasons.push(`contains high-priority cue: ${hint}`);
    }
  }

  for (const hint of MEDIUM_PRIORITY_HINTS) {
    if (cleaned.includes(hint)) {
      score += 1;
      reasons.push(`contains medium-priority cue: ${hint}`);
    }
  }

  if (/\b\d{1,2}:\d{2}\b/.test(cleaned)) {
    score += 1;
    reasons.push("contains explicit time");
  }

  if (/\b(today|tomorrow|friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b/.test(cleaned)) {
    score += 1;
    reasons.push("contains schedule cue");
  }

  let priority = "low";
  if (score >= 6) {
    priority = "high";
  } else if (score >= 2) {
    priority = "medium";
  }

  return {
    priority,
    score,
    reasons
  };
}

export function routeWorkflow(text) {
  const cleaned = normalizeWhitespace(text).toLowerCase();

  const routingRules = [
    { route: "incidentResponse", keywords: ["outage", "incident", "broken", "failure", "production", "down"] },
    { route: "taskPlanning", keywords: ["todo", "action item", "plan", "task", "milestone", "deliverable"] },
    { route: "communicationDraft", keywords: ["reply", "email", "respond", "draft", "message"] },
    { route: "meetingFollowup", keywords: ["meeting", "notes", "follow up", "follow-up", "summary"] },
    { route: "documentation", keywords: ["docs", "documentation", "guide", "readme", "manual"] }
  ];

  for (const rule of routingRules) {
    if (rule.keywords.some((keyword) => cleaned.includes(keyword))) {
      return { route: rule.route };
    }
  }

  return { route: "generalAnalysis" };
}

export function parseContent(content) {
  const html = String(content ?? "");
  const $ = cheerio.load(html);

  const title = $("title").first().text().trim() || null;

  const headings = [];
  $("h1, h2, h3").each((_, element) => {
    headings.push($(element).text().replace(/\s+/g, " ").trim());
  });

  const links = [];
  $("a[href]").each((_, element) => {
    links.push({
      text: $(element).text().replace(/\s+/g, " ").trim(),
      href: $(element).attr("href")
    });
  });

  let plainText = $.root().text().replace(/\s+/g, " ").trim();
  if (!plainText) {
    plainText = normalizeWhitespace(html);
  }

  const sections = [];
  $("h1, h2, h3").each((_, element) => {
    const heading = $(element).text().replace(/\s+/g, " ").trim();
    const contentParts = [];
    let sibling = $(element).next();

    while (
      sibling.length > 0 &&
      !["h1", "h2", "h3"].includes((sibling.get(0)?.tagName || "").toLowerCase())
    ) {
      contentParts.push(sibling.text().replace(/\s+/g, " ").trim());
      sibling = sibling.next();
    }

    sections.push({
      heading,
      content: contentParts.join(" ").replace(/\s+/g, " ").trim()
    });
  });

  return {
    title,
    headings,
    links,
    plainText,
    sections
  };
}

export function resolveDatetime(text) {
  const cleaned = normalizeWhitespace(text);
  if (!cleaned) {
    return { error: "No datetime text provided." };
  }

  const parsed = chrono.parseDate(cleaned);
  if (!parsed) {
    return { error: "Could not parse datetime." };
  }

  return {
    original: cleaned,
    resolvedIso: parsed.toISOString(),
    humanReadable: parsed.toLocaleString()
  };
}

export function formatJson(value) {
  const parsed = safeParseJsonOrString(value);

  try {
    return {
      formattedJson: JSON.stringify(parsed, null, 2)
    };
  } catch (error) {
    return {
      error: `Could not format JSON: ${error.message}`
    };
  }
}

function ensureAllowedExpression(expression) {
  if (!/^[0-9+\-*/().,\s_a-zA-Z]+$/.test(expression)) {
    throw new Error("Expression contains unsupported characters.");
  }
}

export function calculator(expression) {
  const cleaned = normalizeWhitespace(expression);
  if (!cleaned) {
    return { error: "No expression provided." };
  }

  const allowedMath = {
    sqrt: Math.sqrt,
    sin: Math.sin,
    cos: Math.cos,
    tan: Math.tan,
    log: Math.log,
    log10: Math.log10,
    exp: Math.exp,
    abs: Math.abs,
    ceil: Math.ceil,
    floor: Math.floor,
    pi: Math.PI,
    e: Math.E
  };

  try {
    ensureAllowedExpression(cleaned);

    const evaluator = new Function(
      ...Object.keys(allowedMath),
      `"use strict"; return (${cleaned});`
    );

    const result = evaluator(...Object.values(allowedMath));

    if (typeof result !== "number" || Number.isNaN(result)) {
      throw new Error("Expression did not produce a valid numeric result.");
    }

    return {
      expression: cleaned,
      result
    };
  } catch (error) {
    return {
      error: `Could not evaluate expression: ${error.message}`
    };
  }
}

export function analyzeText(text) {
  const cleaned = normalizeWhitespace(text);
  if (!cleaned) {
    return {
      summary: "",
      keywords: [],
      tasks: [],
      priority: { priority: "low", score: 0, reasons: [] },
      route: "generalAnalysis",
      nextSteps: []
    };
  }

  const summaryResult = summarizeText(cleaned);
  const keywordsResult = extractKeywords(cleaned);
  const tasksResult = extractTasks(cleaned);
  const priorityResult = scorePriority(cleaned);
  const routeResult = routeWorkflow(cleaned);

  const nextSteps = [];
  if (tasksResult.tasks.length > 0) {
    for (const task of tasksResult.tasks.slice(0, 3)) {
      nextSteps.push(`Complete task: ${task.task}`);
    }
  } else {
    nextSteps.push("Review the summary and confirm the intended action.");
    nextSteps.push("Clarify ownership and deadline if missing.");
  }

  return {
    summary: summaryResult.summary,
    keywords: keywordsResult.keywords,
    tasks: tasksResult.tasks,
    priority: priorityResult,
    route: routeResult.route,
    nextSteps
  };
}

export const tools = {
  summarizeText,
  extractKeywords,
  extractTasks,
  scorePriority,
  routeWorkflow,
  parseContent,
  resolveDatetime,
  formatJson,
  calculator,
  analyzeText
};

export const toolDescriptions = {
  summarizeText: "Create a short deterministic summary of input text.",
  extractKeywords: "Extract top keywords from input text.",
  extractTasks: "Extract likely action items or tasks from text.",
  scorePriority: "Estimate priority level from urgency and blocker cues.",
  routeWorkflow: "Route input into a likely workflow category.",
  parseContent: "Parse HTML or text into normalized sections, headings, links, and plain text.",
  resolveDatetime: "Resolve date/time phrases into ISO and human-readable values.",
  formatJson: "Pretty-format JSON-compatible input.",
  calculator: "Safely evaluate arithmetic expressions.",
  analyzeText: "Run a combined text analysis including summary, keywords, tasks, priority, and next steps."
};

export function listTools() {
  return {
    tools: Object.keys(tools).map((name) => ({
      name,
      description: toolDescriptions[name]
    }))
  };
}

export function buildToolsPrompt() {
  const lines = [
    "You can use the following local deterministic tools:",
    ""
  ];

  for (const [name, description] of Object.entries(toolDescriptions)) {
    lines.push(`- ${name}: ${description}`);
  }

  lines.push(
    "",
    "When a tool is needed, respond in this format:",
    "TOOL: <toolName>",
    "INPUT: <toolInput>",
    "",
    "After receiving the tool result, continue your reasoning using the observation."
  );

  return lines.join("\n");
}

export function runTool(name, inputData) {
  const tool = tools[name];
  if (!tool) {
    return { error: `Tool '${name}' not found.` };
  }

  try {
    return tool(inputData);
  } catch (error) {
    return {
      error: `Tool '${name}' failed: ${error.message}`
    };
  }
}
