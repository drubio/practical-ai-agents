import 'dotenv/config';
import readline from 'node:readline';
import { ALL_MODEL_IDENTIFIERS, getIdentifierMappings } from '../llm_models.mjs';
import {
  compactModelSelectionLines,
  getAllProviders,
  getApiKey,
  getChapterLogger,
  getDisplayName,
  logToolCall,
  modelIdentifiersForProviders,
  normalizeResponseText,
  selectProviderModelIdentifier,
  sortProvidersByDisplayOrder,
  closeSharedAsk,
} from '../utils.mjs';
import { interactiveBasicQuestionLoop } from '../essentials/utils.mjs';

export { ALL_MODEL_IDENTIFIERS, compactModelSelectionLines, getChapterLogger, getIdentifierMappings, logToolCall };

export function buildCommonArgs(argv = process.argv.slice(2)) {
  let mode = 'cli';
  let stream = false;
  let host = '0.0.0.0';
  let port = Number(process.env.PORT || 8000);
  let modelIdentifier = null;
  let temperature = 0.7;
  let maxTokens = 1000;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === 'cli' || arg === 'web') mode = arg;
    else if (arg === '--stream') stream = true;
    else if (arg === '--host') host = argv[++i];
    else if (arg === '--port') port = Number(argv[++i]);
    else if (arg === '--model-identifier') modelIdentifier = argv[++i];
    else if (arg === '--temperature') temperature = Number(argv[++i]);
    else if (arg === '--max-tokens') maxTokens = Number(argv[++i]);
  }
  return { mode, stream, host, port, modelIdentifier, temperature, maxTokens };
}

function askOnce(prompt) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve) => {
    rl.question(prompt, (answer) => {
      rl.close();
      resolve(answer);
    });
  });
}


export async function selectStartupModel(modelIdentifiers, mode, explicitModelIdentifier) {
  if (explicitModelIdentifier) return explicitModelIdentifier;
  const ids = (modelIdentifiers && modelIdentifiers.length ? modelIdentifiers : ALL_MODEL_IDENTIFIERS);
  if (!ids.length) throw new Error('No model identifiers configured');

  const providerStatuses = Object.fromEntries(getAllProviders().map((provider) => [
    provider,
    getApiKey(provider) ? '✓ API key configured' : '✗ API key not found',
  ]));
  const availableProviders = sortProvidersByDisplayOrder(
    Object.entries(providerStatuses).filter(([, status]) => status.startsWith('✓')).map(([provider]) => provider),
  );
  const availableModelIds = modelIdentifiersForProviders(availableProviders).filter((identifier) => ids.includes(identifier));

  if (mode !== 'cli' || !process.stdin.isTTY || !process.stdout.isTTY) return (availableModelIds.length ? availableModelIds : ids)[0];

  console.log('\n=== LangChain Framework - Provider Status ===');
  for (const [provider, message] of Object.entries(providerStatuses)) console.log(`${getDisplayName(provider)}: ${message}`);
  console.log(`${'='.repeat(50)}\n`);

  if (!availableModelIds.length) {
    console.log('No models available for initialized providers; using the first configured model.');
    return ids[0];
  }

  const selectedModel = await selectProviderModelIdentifier(availableProviders, askOnce);
  const modelConfig = getIdentifierMappings()[selectedModel];
  console.log(
    '\nUsing model: '
    + `${getDisplayName(modelConfig.provider)} `
    + `(provider: ${modelConfig.provider}, `
    + `model: ${modelConfig.model} / `
    + `${modelConfig.name} / `
    + `${modelConfig.tier})`,
  );
  return selectedModel;
}

export function extractTextContent(content) {
  if (typeof content === 'string') return normalizeResponseText(content);
  if (Array.isArray(content)) {
    return content.map((item) => (typeof item === 'string' ? item : (item?.text || item?.content || ''))).join('');
  }
  if (content && typeof content === 'object') return content.text || content.content || normalizeResponseText(content);
  return normalizeResponseText(content);
}

export function langChainMessageTypeName(message) {
  return message?.constructor?.name || 'Message';
}

export function langChainMessageToolCalls(message) {
  return Array.isArray(message?.tool_calls) && message.tool_calls.length ? message.tool_calls : null;
}

export function langChainStreamChunkFromEvent(event) {
  return Array.isArray(event) ? event[1]?.[0] ?? event[1] : event;
}

export async function runMode(manager, mode, host = '0.0.0.0', port = 8000, stream = false) {
  if (mode === 'web') {
    const { runWebServer } = await import('./web.js');
    await runWebServer(manager, host, port, stream);
    return;
  }
  console.log(`\n===== ${manager.framework} CLI =====`);
  const names = Array.isArray(manager.toolNames) ? manager.toolNames : [];
  if (names.length) {
    console.log('Available local tools:');
    names.forEach((n) => console.log(`  - ${n}`));
  } else {
    console.log('Available local tools: (none declared)');
  }
  console.log(`\n${manager.toolTriggerHelp || 'Tools are triggered automatically from your prompt.'}`);
  console.log('====================================');
  try {
    await interactiveBasicQuestionLoop(manager, {
      provider: manager.provider,
      modelIdentifier: manager.modelIdentifier,
      askQuestion: (prompt) => manager.askQuestion(prompt, { stream }),
    });
  } finally {
    closeSharedAsk();
  }
}
