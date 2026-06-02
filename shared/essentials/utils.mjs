import { AsyncLocalStorage } from "node:async_hooks";

import {
  formatFilename,
  getAllProviderNames,
  getDefaultModel as baseGetDefaultModel,
  getNonEmptyInput,
  normalizeResponseText,
  parseStructuredJsonResponse,
  saveResponseToFile,
  closeSharedAsk,
  displayProviderResponse,
  formatProviderSummary,
  getAllProviders,
  getApiKey,
  getDefaultModelDetails,
  getDisplayName,
  getSharedAsk,
  getUserChoice,
  getUserParameters,
  modelIdentifiersForProviders,
  selectModelIdentifier,
  selectProviderModelIdentifier,
  compactModelSelectionLines,
  sortProvidersByDisplayOrder,
} from "../utils.mjs";
import { getIdentifierMappings } from "../llm_models.mjs";

export {
  displayProviderResponse,
  formatFilename,
  formatProviderSummary,
  getAllProviderNames,
  getAllProviders,
  getApiKey,
  getDefaultModelDetails,
  getDisplayName,
  getNonEmptyInput,
  getUserChoice,
  getUserParameters,
  modelIdentifiersForProviders,
  selectModelIdentifier,
  selectProviderModelIdentifier,
  compactModelSelectionLines,
  normalizeResponseText,
  parseStructuredJsonResponse,
  saveResponseToFile,
  sortProvidersByDisplayOrder,
};


const selectedModelStorage = new AsyncLocalStorage();

export function getDefaultModel(provider) {
  const selectedIdentifier = selectedModelStorage.getStore() || null;
  if (selectedIdentifier) {
    const config = getIdentifierMappings()[selectedIdentifier];
    if (config?.provider === provider) return config.model;
  }
  return baseGetDefaultModel(provider);
}

export function withSelectedModelIdentifier(modelIdentifier, fn) {
  return selectedModelStorage.run(modelIdentifier || null, fn);
}

export function getModelIdentifierConfig(modelIdentifier) {
  if (modelIdentifier === null || typeof modelIdentifier === "undefined") return null;
  return getIdentifierMappings()[String(modelIdentifier).trim()] || null;
}

export class EssentialsLLMManager {
  constructor(frameworkName) {
    this.framework = frameworkName;
    this.initializationMessages = {};
  }

  async _checkProviders() {
    for (const provider of getAllProviders()) {
      if (getApiKey(provider)) {
        try {
          await this._testProvider(provider);
          this.initializationMessages[provider] = "✓ Initialized successfully";
        } catch (error) {
          this.initializationMessages[provider] = `✗ Failed: ${error.message}`;
        }
      } else {
        this.initializationMessages[provider] = "✗ API key not found";
      }
    }
  }

  async _testProvider() {
    throw new Error("Subclasses must implement _testProvider");
  }

  getAvailableProviders() {
    return Object.entries(this.initializationMessages)
      .filter(([, status]) => status.startsWith("✓"))
      .map(([provider]) => provider);
  }

  displayInitializationStatus() {
    console.log(`\n=== ${this.framework} Framework - Provider Status ===`);
    for (const [provider, message] of Object.entries(this.initializationMessages)) {
      console.log(`${getDisplayName(provider)}: ${message}`);
    }
    console.log(`${"=".repeat(50)}\n`);
  }
}

export function managerSupportsInteractiveMemory(manager) {
  const fullMemorySupported = manager?.memoryEnabled === true
    && typeof manager.askQuestion === "function"
    && typeof manager.getHistory === "function"
    && typeof manager.resetMemory === "function";
  const retrievalMemorySupported = manager?.retrievalMemoryEnabled === true
    && typeof manager.askQuestion === "function"
    && typeof manager.getHistory === "function"
    && typeof manager.resetMemory === "function";
  return fullMemorySupported || retrievalMemorySupported;
}


export async function interactiveBasicQuestionLoop(manager, { provider = null, modelIdentifier = null, askQuestion = null, prompt = "\nAsk a question (or 'exit'): " } = {}) {
  const ask = getSharedAsk();
  while (true) {
    const userInput = (await ask(prompt)).trim();
    if (["exit", "quit"].includes(userInput.toLowerCase())) {
      console.log("Exiting.");
      break;
    }
    if (!userInput) {
      console.log("Input cannot be empty. Please try again.");
      continue;
    }
    const response = await withSelectedModelIdentifier(modelIdentifier, async () => (askQuestion ? askQuestion(userInput) : manager.askQuestion(userInput, provider)));
    if (!manager.printsOwnOutput) displayProviderResponse(provider || manager.provider || "unknown", response, manager.framework);
  }
}

export async function interactiveCli(manager, modelIdentifier = null) {
  const ask = getSharedAsk();
  try {
    console.log("=".repeat(60));
    console.log(`Agent Application - ${manager.framework} Framework`);
    console.log("=".repeat(60));

    manager.displayInitializationStatus();
    let availableProviders = manager.getAvailableProviders();
    if (availableProviders.length === 0) {
      console.log("No providers available. Check your .env file.");
      return;
    }

    const { temperature, maxTokens } = await getUserParameters(ask);
    console.log(`\nUsing temperature: ${temperature}, max tokens: ${maxTokens}`);
    availableProviders = sortProvidersByDisplayOrder(availableProviders);
    //console.log("\nAvailable providers:");
    //for (const providerName of availableProviders) {
    //  console.log(`- ${formatProviderSummary(providerName)}`);
    //}

    const availableModelIdentifiers = modelIdentifiersForProviders(availableProviders);
    if (availableModelIdentifiers.length === 0) {
      console.log("No models available for initialized providers.");
      return;
    }

    const memorySupported = managerSupportsInteractiveMemory(manager);

    if (modelIdentifier) {
      if (!availableModelIdentifiers.includes(modelIdentifier)) {
        console.log(`Model identifier '${modelIdentifier}' is not available for initialized providers.`);
        return;
      }
    } else {
      modelIdentifier = await selectProviderModelIdentifier(availableProviders, ask);
    }
    const modelConfig = getIdentifierMappings()[modelIdentifier];
    const provider = modelConfig.provider;
    const selectedProviderDetails = getDefaultModelDetails(provider);
    console.log(
      "\nUsing model: "
      + `${selectedProviderDetails.displayName} `
      + `(provider: ${modelConfig.provider}, `
      + `model: ${modelConfig.model} / `
      + `${modelConfig.name} / `
      + `${modelConfig.tier})`,
    );
    let sessionId = "default";
    if (memorySupported) {
      const sessionIdInput = (await ask("Enter memory session ID (default: 'default'): ")).trim();
      if (sessionIdInput) sessionId = sessionIdInput;
      console.log(`Using memory session: ${sessionId}`);
    }
    console.log(`\n${"=".repeat(50)}`);
    console.log(`${manager.framework.toUpperCase()} INTERACTIVE MODE - ${getDisplayName(provider).toUpperCase()}`);
    console.log("=".repeat(50));

    if (!memorySupported) {
      await interactiveBasicQuestionLoop(manager, {
        provider,
        modelIdentifier,
        askQuestion: (userInput) => manager.askQuestion(userInput, provider, "{topic}", maxTokens, temperature),
      });
      console.log(`\nThank you for using the ${manager.framework} Agent Application!`);
      return;
    }

    while (true) {
      const prompt = memorySupported ? "\nAsk a question (or 'history', 'clear', 'exit'): " : "\nAsk a question (or 'exit'): ";
      const userInput = (await ask(prompt)).trim();
      if (["exit", "quit"].includes(userInput.toLowerCase())) {
        console.log("Exiting.");
        break;
      }
      if (!userInput) {
        console.log("Input cannot be empty. Please try again.");
        continue;
      }
      if (userInput.toLowerCase() === "history") {
        if (memorySupported && typeof manager.getHistory === "function") {
          const history = await Promise.resolve(manager.getHistory(provider, sessionId));
          console.log(`\n🧠 Memory for ${getDisplayName(provider)} (session: ${sessionId}):`);
          for (const turn of history.turns || []) {
            console.log(`[${String(turn.role).charAt(0).toUpperCase()}${String(turn.role).slice(1)}] ${turn.content}`);
          }
          if (!history.turns || history.turns.length === 0) console.log("No memory yet.");
        } else {
          console.log("⚠️ This manager does not support memory history.");
        }
      } else if (userInput.toLowerCase() === "clear") {
        if (memorySupported && typeof manager.resetMemory === "function") {
          await Promise.resolve(manager.resetMemory(provider, sessionId));
          console.log(`✅ Memory cleared for session '${sessionId}'`);
        } else {
          console.log("⚠️ This manager does not support memory reset.");
        }
      } else {
        const response = await withSelectedModelIdentifier(modelIdentifier, async () => (memorySupported
          ? manager.askQuestion(userInput, provider, "{topic}", maxTokens, temperature, sessionId)
          : manager.askQuestion(userInput, provider, "{topic}", maxTokens, temperature)));
        displayProviderResponse(provider, response, manager.framework);
      }
    }

    console.log(`\nThank you for using the ${manager.framework} Agent Application!`);
  } finally {
    closeSharedAsk();
  }
}

// Backwards-compatible alias for chapter code that imports BaseLLMManager.
export const BaseLLMManager = EssentialsLLMManager;
