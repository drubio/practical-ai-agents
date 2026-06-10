import { randomUUID } from 'node:crypto';

import { tool } from '@langchain/core/tools';
import { z } from 'zod';

export function calculator(expression) {
  const text = String(expression ?? '').trim();
  if (!text) return { error: 'No expression provided.' };

  if (!/^[\d\s+\-*/().%^]+$/.test(text)) {
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
    human_readable: date.toUTCString(),
  };
}

export function generateUUID() {
  return { uuid: randomUUID() };
}


export function createCalculatorTool(pendingToolLogs) {
  return tool(({ expression }) => {
    const input = { expression };
    const output = calculator(expression);
    pendingToolLogs.push({ name: 'calculator', input, output });
    return output;
  }, {
    name: 'calculator',
    description: 'Evaluate a basic arithmetic expression.',
    schema: z.object({
      expression: z.string().describe('Arithmetic expression to evaluate.'),
    }),
  });
}

export function createGenerateUuidTool(pendingToolLogs) {
  return tool(() => {
    const input = {};
    const output = generateUUID();
    pendingToolLogs.push({ name: 'generate_uuid', input, output });
    return output;
  }, {
    name: 'generate_uuid',
    description: 'Generate a unique UUID identifier. No input is required.',
    schema: z.object({}).strict(),
  });
}
