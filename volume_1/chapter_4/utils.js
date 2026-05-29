/** utils.js - Common utilities and configurations shared across all JavaScript frameworks. */

import {
  closeSharedAsk,
  displayProviderResponse,
  formatFilename,
  formatProviderSummary,
  getAllProviderNames,
  getAllProviders,
  getApiKey,
  getDefaultModel,
  getDefaultModelDetails,
  getDisplayName,
  getNonEmptyInput,
  getSharedAsk,
  getUserChoice,
  getUserParameters,
  parseStructuredJsonResponse,
  saveResponseToFile,
} from '../../shared/utils.mjs';

export {
  displayProviderResponse,
  formatFilename,
  formatProviderSummary,
  getAllProviderNames,
  getAllProviders,
  getApiKey,
  getDefaultModel,
  getDefaultModelDetails,
  getDisplayName,
  getNonEmptyInput,
  getUserChoice,
  getUserParameters,
  parseStructuredJsonResponse,
  saveResponseToFile,
};

export class BaseLLMManager {
  constructor(frameworkName) {
    this.framework = frameworkName;
    this.initializationMessages = {};
  }

  async _checkProviders() {
    for (const provider of getAllProviders()) {
      if (getApiKey(provider)) {
        try {
          await this._testProvider(provider);
          this.initializationMessages[provider] = '✓ Initialized successfully';
        } catch (error) {
          this.initializationMessages[provider] = `✗ Failed: ${error.message}`;
        }
      } else {
        this.initializationMessages[provider] = '✗ API key not found';
      }
    }
  }

  getAvailableProviders() {
    return Object.entries(this.initializationMessages).filter(([_, status]) => status.startsWith('✓')).map(([provider]) => provider);
  }

  displayInitializationStatus() {
    console.log(`\n=== ${this.framework} Framework - Provider Status ===`);
    for (const [provider, message] of Object.entries(this.initializationMessages)) {
      console.log(`${getDisplayName(provider)}: ${message}`);
    }
    console.log('='.repeat(50) + '\n');
  }
}

export async function interactiveCli(manager) {
  const ask = getSharedAsk();
  try {
    console.log('='.repeat(60));
    console.log(`Agent Application - ${manager.framework} Framework`);
    console.log('='.repeat(60));
    manager.displayInitializationStatus();
    const availableProviders = manager.getAvailableProviders();
    if (availableProviders.length === 0) {
      console.log('No providers available. Check your .env file.');
      return;
    }
    const { temperature, maxTokens } = await getUserParameters(ask);
    console.log(`\nUsing temperature: ${temperature}, max tokens: ${maxTokens}`);
    const sortedProviders = [...availableProviders].sort((a, b) => (a === 'openai' ? -1 : b === 'openai' ? 1 : getDisplayName(a).localeCompare(getDisplayName(b))));
    console.log('\nAvailable providers:');
    sortedProviders.forEach((provider) => console.log(`- ${formatProviderSummary(provider)}`));
    const memorySupported = Boolean((manager.memoryEnabled || manager.retrievalMemoryEnabled) && typeof manager.askQuestion === 'function' && typeof manager.getHistory === 'function' && typeof manager.resetMemory === 'function');

    const provider = sortedProviders[await getUserChoice(sortedProviders.map((p) => getDisplayName(p)), 'Select a provider:', ask)];
    const providerDetails = getDefaultModelDetails(provider);
    console.log(`\nUsing provider: ${providerDetails.displayName} (provider: ${providerDetails.canonicalProvider}, default model: ${providerDetails.defaultModel} / ${providerDetails.defaultModelIdentifier} / ${providerDetails.defaultModelTier})`);
    let sessionId = 'default';
    if (memorySupported) {
      const sessionInput = (await ask("Enter memory session ID (default: 'default'): ")).trim();
      if (sessionInput) sessionId = sessionInput;
      console.log(`Using memory session: ${sessionId}`);
    }
    console.log('\n' + '='.repeat(50));
    console.log(`${manager.framework.toUpperCase()} INTERACTIVE MODE - ${getDisplayName(provider).toUpperCase()}`);
    console.log('='.repeat(50));
    while (true) {
      const userInput = (await ask(memorySupported ? "\nAsk a question (or 'history', 'clear', 'exit'): " : "\nAsk a question (or 'exit'): ")).trim();
      if (['exit', 'quit'].includes(userInput.toLowerCase())) break;
      if (!userInput) {
        console.log('Input cannot be empty. Please try again.');
        continue;
      }
      if (memorySupported && userInput.toLowerCase() === 'history') {
        const history = await Promise.resolve(manager.getHistory(provider, sessionId));
        console.log(`\n🧠 Memory for ${getDisplayName(provider)} (session: ${sessionId}):`);
        for (const turn of history.turns) console.log(`[${String(turn.role || 'unknown').replace(/^./, (c) => c.toUpperCase())}] ${turn.content}`);
        if (!history.turns.length) console.log('No memory yet.');
        continue;
      }
      if (memorySupported && userInput.toLowerCase() === 'clear') {
        await Promise.resolve(manager.resetMemory(provider, sessionId));
        console.log(`✅ Memory cleared for session '${sessionId}'`);
        continue;
      }
      const result = memorySupported ? await manager.askQuestion(userInput, provider, '{topic}', maxTokens, temperature, sessionId) : await manager.askQuestion(userInput, provider, '{topic}', maxTokens, temperature);
      displayProviderResponse(provider, result, manager.framework);
    }
  } catch (error) {
    console.error('Fatal Error:', error);
  } finally {
    console.log(`\nThank you for using the ${manager.framework} Agent Application!`);
    closeSharedAsk();
  }
}
