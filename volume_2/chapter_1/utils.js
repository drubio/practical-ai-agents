import 'dotenv/config';
import readline from 'node:readline';
import { ALL_MODEL_IDENTIFIERS, getIdentifierMappings } from '../../shared/llm_models.mjs';

export function getChapterLogger(name) {
  return { info: (...a) => console.log(`[INFO] [${name}]`, ...a), error: (...a) => console.error(`[ERROR] [${name}]`, ...a) };
}

export function logToolCall(logger, toolName, fn) {
  return (input) => { logger.info(`tool=${toolName} input=`, input); const out = fn(input); logger.info(`tool=${toolName} output=`, out); return out; };
}

export function buildCommonArgs(argv = process.argv.slice(2)) {
  let mode = 'cli', stream = false, host = '0.0.0.0', port = Number(process.env.PORT || 8000), modelIdentifier = null;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === 'cli' || arg === 'web') mode = arg;
    else if (arg === '--stream') stream = true;
    else if (arg === '--host') host = argv[++i];
    else if (arg === '--port') port = Number(argv[++i]);
    else if (arg === '--model-identifier') modelIdentifier = argv[++i];
  }
  return { mode, stream, host, port, modelIdentifier };
}

function chooseModelInteractive(ids) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  console.log('\nModel selection:');
  ids.forEach((id, i) => console.log(`${i + 1}. ${id}${i === 0 ? ' [default]' : ''}`));
  return new Promise((resolve) => {
    const ask = () => rl.question(`Select model (1-${ids.length}, default 1): `, (raw) => {
      const t = String(raw || '').trim();
      if (!t) { rl.close(); return resolve(ids[0]); }
      const n = Number(t);
      if (Number.isInteger(n) && n >= 1 && n <= ids.length) { rl.close(); return resolve(ids[n - 1]); }
      console.log('Invalid selection. Try again.');
      ask();
    });
    ask();
  });
}

export async function selectStartupModel(modelIdentifiers, mode, explicitModelIdentifier) {
  if (explicitModelIdentifier) return explicitModelIdentifier;
  const ids = (modelIdentifiers && modelIdentifiers.length ? modelIdentifiers : ALL_MODEL_IDENTIFIERS);
  if (mode !== 'cli' || !process.stdin.isTTY || !process.stdout.isTTY) return ids[0];
  return chooseModelInteractive(ids);
}

export async function runMode(manager, mode, host = '0.0.0.0', port = 8000, stream = false) {
  if (mode === 'web') {
    const { runWebServer } = await import('./web.js');
    await runWebServer(manager, host, port, stream);
    return;
  }
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  console.log(`\n===== ${manager.framework} CLI =====`);
  console.log("Type a question and press Enter.");
  console.log("Type 'exit' to quit.\n");
  const names = Array.isArray(manager.toolNames) ? manager.toolNames : [];
  if (names.length) {
    console.log('Available local tools:');
    names.forEach((n) => console.log(`  - ${n}`));
  } else {
    console.log('Available local tools: (none declared)');
  }
  console.log(`\n${manager.toolTriggerHelp || 'Tools are triggered automatically from your prompt.'}`);
  console.log('Tip: ask for a UUID to force tool usage.');
  console.log('====================================');
  while (true) {
    const prompt = await new Promise((resolve) => rl.question('> ', resolve));
    if (!prompt || prompt.trim().toLowerCase() === 'exit') break;
    const result = await manager.askQuestion(prompt.trim());
    console.log(result.success ? result.finalText : result.error);
  }
  rl.close();
}

export { ALL_MODEL_IDENTIFIERS, getIdentifierMappings };
