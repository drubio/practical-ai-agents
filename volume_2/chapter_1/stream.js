/** Streaming helpers for volume 2 chapter agent APIs. */

export function normalizeResponseText(payload) {
  if (payload == null) return "";

  if (typeof payload === "string") {
    return payload;
  }

  if (typeof payload === "object") {
    for (const key of ["response", "answer", "content", "text", "message"]) {
      const value = payload[key];
      if (typeof value === "string") return value;
    }
  }

  return String(payload);
}

export function chunkText(text, chunkSize = 28) {
  const clean = text || "";
  if (!clean) return [""];

  const chunks = [];
  for (let index = 0; index < clean.length; index += chunkSize) {
    chunks.push(clean.slice(index, index + chunkSize));
  }
  return chunks;
}

export async function* iterTextChunks(text, chunkSize = 28, delayMs = 0) {
  for (const part of chunkText(text, chunkSize)) {
    if (delayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    yield part;
  }
}

export function toSseLine(data) {
  return `data: ${JSON.stringify(data)}\n\n`;
}
