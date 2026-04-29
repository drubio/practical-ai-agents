import { randomUUID } from 'node:crypto';

export function calculator(expression) {
  const text = String(expression ?? '').trim();
  if (!text) return { error: 'No expression provided.' };

  if (!/^[\d\s+\-*/().%]+$/.test(text)) {
    return { error: 'Only basic arithmetic characters are allowed.' };
  }

  try {
    const result = Function(`"use strict"; return (${text.replace(/\^/g, '**')});`)();
    if (typeof result !== 'number' || Number.isNaN(result)) return { error: 'Invalid numeric result.' };
    return { expression: text, result };
  } catch (error) {
    return { error: `Could not evaluate expression: ${error.message}` };
  }
}

export function resolveDatetime(text) {
  const value = String(text ?? '').trim();
  if (!value) return { error: 'No datetime text provided.' };
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return { error: 'Could not parse datetime.' };
  return {
    original: value,
    resolved_iso: date.toISOString(),
    human_readable: date.toUTCString()
  };
}

export function generateUUID() {
  return { uuid: randomUUID() };
}
