"""
Common utilities and configurations shared across all frameworks
"""

import os
import ast
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from stream import normalize_response_text

# Load environment variables
load_dotenv()


# Provider configurations
PROVIDERS = {
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-5",
        "display_name": "Anthropic Claude"
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-5.2",
        "display_name": "OpenAI GPT"
    },
    "google": {
        "api_key_env": "GOOGLE_API_KEY",
        "default_model": "gemini-3-flash-preview",
        "display_name": "Google Gemini"
    },
    "xai": {
        "api_key_env": "XAI_API_KEY",
        "default_model": "grok-4",
        "display_name": "xAI Grok"
    }
}

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider"""
    if provider in PROVIDERS:
        return os.getenv(PROVIDERS[provider]["api_key_env"])
    return None

def get_default_model(provider: str) -> str:
    """Get default model for a provider"""
    return PROVIDERS.get(provider, {}).get("default_model", "")

def get_display_name(provider: str) -> str:
    """Get display name for a provider"""
    return PROVIDERS.get(provider, {}).get("display_name", provider.capitalize())

def get_all_providers() -> List[str]:
    """Get list of all configured providers"""
    return list(PROVIDERS.keys())

def get_available_providers() -> List[str]:
    """Get list of providers with available API keys"""
    available = []
    for provider_name in PROVIDERS.keys():
        if get_api_key(provider_name):
            available.append(provider_name)
    return available


def parse_structured_json_response(raw: Any) -> Dict[str, Any]:
    """Parse structured model output into a JSON object with tolerant fallbacks."""
    if raw is None:
        content = ""
    elif isinstance(raw, str):
        content = raw.strip()
    elif isinstance(raw, dict):
        content = json.dumps(raw, ensure_ascii=False)
    elif hasattr(raw, "content") and isinstance(raw.content, str):
        content = raw.content.strip()
    else:
        content = str(raw).strip()

    content_match = re.search(
        r'content=(["\'])((?:\\.|(?!\1).)*)\1\s+additional_kwargs=',
        content,
        flags=re.DOTALL,
    )
    if content_match:
        raw_quoted_content = f"{content_match.group(1)}{content_match.group(2)}{content_match.group(1)}"
        try:
            decoded = ast.literal_eval(raw_quoted_content)
            if isinstance(decoded, str):
                content = decoded.strip()
        except Exception:
            pass

    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Some SDKs return list-like content blocks as Python repr strings.
    if content.startswith("[") and content.endswith("]"):
        try:
            blocks = ast.literal_eval(content)
            if isinstance(blocks, list):
                for block in blocks:
                    if isinstance(block, dict):
                        maybe_text = block.get("text") or block.get("content")
                        if isinstance(maybe_text, str) and maybe_text.strip():
                            content = maybe_text.strip()
                            if content.startswith("```json"):
                                content = content[7:]
                            if content.startswith("```"):
                                content = content[3:]
                            if content.endswith("```"):
                                content = content[:-3]
                            content = content.strip()
                            break
        except Exception:
            pass

    # Fallback: extract first complete JSON object from mixed wrapper text.
    start = content.find("{")
    if start == -1:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Parsed structured content is not a JSON object")

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(content)):
        ch = content[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(content[start : idx + 1])
                if isinstance(parsed, dict):
                    return parsed
                raise ValueError("Parsed structured content is not a JSON object")

    parsed = json.loads(content)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("Parsed structured content is not a JSON object")

def get_user_parameters():
    """Get temperature and max_tokens from user input with validation"""
    # Ask for temperature setting
    temp_input = input("Temperature (0.0-2.0, default 0.7): ").strip()
    try:
        temperature = float(temp_input) if temp_input else 0.7
        temperature = max(0.0, min(2.0, temperature))  # Clamp between 0.0 and 2.0
    except ValueError:
        temperature = 0.7
        print(f"Invalid temperature, using default: {temperature}")
    
    # Ask for max tokens
    tokens_input = input("Max tokens (default 1000): ").strip()
    try:
        max_tokens = int(tokens_input) if tokens_input else 1000
        max_tokens = max(1, min(4000, max_tokens))  # Reasonable limits
    except ValueError:
        max_tokens = 1000
        print(f"Invalid max tokens, using default: {max_tokens}")
    
    return temperature, max_tokens

def save_response_to_file(response: Dict[str, Any], filename: str) -> None:
    """Save response data to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(response, f, indent=2, default=str)
    print(f"Response saved to {filename}")

def display_provider_response(provider: str, response: Dict[str, Any], framework: str = "") -> None:
    """Display a provider's response with appropriate formatting"""
    framework_suffix = f" ({framework})" if framework else ""
    provider_display = f"{get_display_name(provider)}{framework_suffix} answered:"
    
    print(f"\n=== {provider_display} ===")
    
    # Show configuration if available
    config_parts = []
    if response.get("temperature") is not None:
        config_parts.append(f"temp: {response['temperature']}")
    if response.get("max_tokens") is not None:
        config_parts.append(f"max_tokens: {response['max_tokens']}")
    if response.get("model"):
        config_parts.append(f"model: {response['model']}")
    
    if config_parts:
        print(f"[{', '.join(config_parts)}]")
    
    if response.get("success"):
        raw = response.get("response", "")
        if isinstance(raw, (dict, list)):
            print(json.dumps(raw, indent=2, ensure_ascii=False))
        elif hasattr(raw, "content"):  # Structured message (like AIMessage)
            print(str(raw.content))
        else:
            print(normalize_response_text(raw))
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    print("=" * 60)

def print_initialization_status(framework: str, messages: Dict[str, str]) -> None:
    """Print provider initialization status"""
    print(f"\n=== {framework} Framework - Provider Status ===")
    for provider, message in messages.items():
        print(f"{get_display_name(provider)}: {message}")
    print("=" * 50 + "\n")

def get_user_choice(options: List[str], prompt: str) -> int:
    """Get user selection from a list of options"""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    while True:
        try:
            raw_choice = input(f"Select an option (1-{len(options)}, default 1): ").strip()
            choice = (int(raw_choice) if raw_choice else 1) - 1
            if 0 <= choice < len(options):
                return choice
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_non_empty_input(prompt: str) -> str:
    """Prompt until user provides non-empty input."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Input cannot be empty. Please try again.")

def format_filename(question: str, framework: str) -> str:
    """Format filename for saving results"""
    safe_question = question[:20].replace(' ', '_').replace('?', '').replace('!', '')
    return f"llm_responses_{framework}_{safe_question}.json"

class BaseLLMManager:
    """Base class for LLM framework managers - handles all generic logic"""
    
    def __init__(self, framework_name: str):
        self.framework = framework_name
        self.initialization_messages = {}
        self._check_providers()
    
    def _check_providers(self):
        """Check which providers are available"""
        for provider in get_all_providers():
            if get_api_key(provider):
                try:
                    self._test_provider(provider)
                    self.initialization_messages[provider] = "✓ Initialized successfully"
                except Exception as e:
                    self.initialization_messages[provider] = f"✗ Failed: {str(e)}"
            else:
                self.initialization_messages[provider] = "✗ API key not found"
    
    def _test_provider(self, provider: str):
        """Test if a provider can be initialized - MUST be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _test_provider")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        available = []
        for provider, status in self.initialization_messages.items():
            if status.startswith("✓"):
                available.append(provider)
        return available
    
    def display_initialization_status(self) -> None:
        """Display provider initialization status"""
        print_initialization_status(self.framework, self.initialization_messages)
    
    def ask_question(self, topic: str, provider: str = None, template: str = "{topic}", 
                     max_tokens: int = 1000, temperature: float = 0.7) -> Dict:
        """Ask a question - MUST be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement ask_question")
    
    def query_all_providers(self, topic: str, template: str = "{topic}", 
                           max_tokens: int = 1000, temperature: float = 0.7) -> Dict:
        """Query all available providers - generic implementation"""
        available_providers = self.get_available_providers()
        
        if not available_providers:
            return {
                "success": False,
                "error": "No providers available",
                "prompt": template.format(topic=topic),
                "responses": {}
            }
        
        responses = {}
        for provider in available_providers:
            print(f"Querying {get_display_name(provider)} via {self.framework}...")
            response = self.ask_question(topic=topic, provider=provider, template=template,
                                       max_tokens=max_tokens, temperature=temperature)
            responses[provider] = response
        
        return {
            "success": True,
            "prompt": template.format(topic=topic),
            "responses": responses
        }

def interactive_cli(manager: BaseLLMManager):
    """Generic interactive CLI with optional memory support"""

    print("=" * 60)
    print(f"LLM Application - {manager.framework} Framework")
    print("=" * 60)

    manager.display_initialization_status()
    available_providers = manager.get_available_providers()

    if not available_providers:
        print("No providers available. Check your .env file.")
        return

    temperature, max_tokens = get_user_parameters()
    print(f"\nUsing temperature: {temperature}, max tokens: {max_tokens}")

    available_providers = sorted(available_providers, key=lambda provider: (provider != "openai", get_display_name(provider)))
    print(f"\nAvailable providers: {', '.join([get_display_name(p) for p in available_providers])}")
    query_all = input("Query ALL providers or select one? (all/one, default one): ").strip().lower() or "one"

    # Determine memory compatibility once (kept in sync with JS logic)
    full_memory_supported = (
        getattr(manager, "memory_enabled", False) is True
        and hasattr(manager, "ask_question")
        and hasattr(manager, "get_history")
        and hasattr(manager, "reset_memory")
    )
    retrieval_memory_supported = (
        getattr(manager, "retrieval_memory_enabled", False) is True
        and hasattr(manager, "ask_question")
        and hasattr(manager, "get_history")
        and hasattr(manager, "reset_memory")
    )
    memory_supported = full_memory_supported or retrieval_memory_supported

    if query_all in ["all", "a", ""]:
        question = get_non_empty_input("Enter your question: ")
        print("\n" + "=" * 50)
        print(f"{manager.framework.upper()} API CALLS - QUERYING ALL PROVIDERS")
        print("=" * 50)

        results = manager.query_all_providers(
            topic=question,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if results["success"]:
            for provider, response in results["responses"].items():
                display_provider_response(provider, response, manager.framework)
        else:
            print(f"Error: {results['error']}")

        save_option = input("\nSave results? (y/n): ").lower()
        if save_option in ["y", "yes"]:
            filename = format_filename(question, manager.framework.lower())
            save_response_to_file(results, filename)

    else:
        # SINGLE PROVIDER INTERACTIVE MODE
        provider_names = [get_display_name(p) for p in available_providers]
        choice_idx = get_user_choice(provider_names, "Select a provider:")
        provider = available_providers[choice_idx]

        print(f"\nUsing provider: {get_display_name(provider)}")

        session_id = "default"
        if memory_supported:
            session_id_input = input("Enter memory session ID (default: 'default'): ").strip()
            if session_id_input:
                session_id = session_id_input
            print(f"Using memory session: {session_id}")

        print("\n" + "=" * 50)
        print(f"{manager.framework.upper()} INTERACTIVE MODE - {get_display_name(provider).upper()}")
        print("=" * 50)

        while True:
            if memory_supported:
                user_input = input("\nAsk a question (or 'history', 'clear', 'exit'): ").strip()
            else:
                user_input = input("\nAsk a question (or 'exit'): ").strip()
                
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting.")
                break

            if not user_input:
                print("Input cannot be empty. Please try again.")
                continue

            elif user_input.lower() == "history":
                if memory_supported and hasattr(manager, "get_history"):
                    history = manager.get_history(provider, session_id)
                    print(f"\n🧠 Memory for {get_display_name(provider)} (session: {session_id}):")
                    for turn in history["turns"]:
                        print(f"[{turn['role'].capitalize()}] {turn['content']}")
                    if not history["turns"]:
                        print("No memory yet.")
                else:
                    print("⚠️ This manager does not support memory history.")

            elif user_input.lower() == "clear":
                if memory_supported and hasattr(manager, "reset_memory"):
                    result = manager.reset_memory(provider, session_id)
                    print(f"✅ Memory cleared for session '{session_id}'")
                else:
                    print("⚠️ This manager does not support memory reset.")

            else:
                if memory_supported:
                    result = manager.ask_question(
                        topic=user_input,
                        provider=provider,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        session_id=session_id
                    )
                else:
                    result = manager.ask_question(
                        topic=user_input,
                        provider=provider,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                display_provider_response(provider, result, manager.framework)

    print(f"\nThank you for using the {manager.framework} LLM Application!")
