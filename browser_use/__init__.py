"""Browser automation library with AI capabilities using LLMs and Playwright.

@public

Browser-Use is an async Python library that implements AI browser driver abilities 
using large language models (LLMs) combined with Playwright for web automation. 
This library provides an intuitive API for browser automation with built-in AI 
capabilities for intelligent web interaction.

The library is async-native, requiring `await` for most operations. For convenience,
a `run_sync()` method is provided on the Agent class for synchronous execution in
scripts and notebooks. All browser operations, LLM calls, and tool executions are
designed for concurrent, non-blocking operation.

The library is designed with performance in mind, using lazy imports to minimize 
startup time by up to 80% (from ~1.5s to ~0.3s). Components are only loaded when 
first accessed, making script startup faster and reducing memory usage for 
specialized use cases that don't need all features.

Key components:
    - Agent: Main AI agent for browser automation tasks
    - BrowserSession/Browser: Browser control and session management 
      (Browser is a recommended alias for BrowserSession)
    - Tools/Controller: Tool registry and control for browser actions
    - DomService: DOM manipulation and element interaction
    - Chat Models: Support for various LLM providers (OpenAI, Anthropic, Google, etc.)
    
Provider Feature Matrix:
    OpenAI (ChatOpenAI):
        - Models: GPT-4, GPT-3.5, o1-preview, o1-mini
        - Tool calling: Yes (function calling)
        - Vision: Yes (GPT-4V)
        - Structured output: Yes (response_format)
        - Max context: 128k tokens (GPT-4 Turbo)
        - Streaming: Yes
        
    Anthropic (ChatAnthropic):
        - Models: Claude 3.5 Sonnet, Claude 3 Haiku, Claude 3 Opus
        - Tool calling: Yes (tools)
        - Vision: Yes (all Claude 3 models)
        - Structured output: Yes (via prompting)
        - Max context: 200k tokens
        - Caching: Yes (prompt caching)
        
    Google (ChatGoogle):
        - Models: Gemini Pro, Gemini Flash
        - Tool calling: Yes
        - Vision: Yes
        - Structured output: Yes
        - Max context: 1M tokens (Gemini 1.5 Pro)
        
    Groq (ChatGroq):
        - Models: Llama 3, Mixtral, Gemma
        - Tool calling: Yes
        - Vision: Limited
        - Speed: Very fast (hardware accelerated)
        - Max context: Varies by model
        
Choosing an LLM:
    For complex reasoning: GPT-4, Claude 3.5 Sonnet, Gemini Pro
    For speed/cost: GPT-3.5 Turbo, Claude 3 Haiku, Gemini Flash, Groq models
    For vision tasks: GPT-4V, Claude 3 models, Gemini Pro
    For long context: Gemini 1.5 Pro (1M), Claude (200k), GPT-4 Turbo (128k)
    For local/private: Ollama with Llama 3, Mixtral via ChatOllama

Architecture:
    Browser-Use builds on top of Playwright, which serves as the underlying browser
    automation engine. Playwright provides cross-browser support (Chrome, Firefox, 
    Safari), reliable element interaction, and network interception capabilities.
    Browser-Use adds AI-driven decision making, automatic element detection, and
    intelligent action planning on top of Playwright's solid foundation.

Example:
    >>> from browser_use import Agent, BrowserSession
    >>> async def main():
    ...     browser = BrowserSession()
    ...     agent = Agent(task="Search for Python documentation", browser=browser)
    ...     await agent.run()
    
    >>> # Using specific LLM providers
    >>> from browser_use import ChatOpenAI
    >>> llm = ChatOpenAI(model="gpt-4")
    >>> agent = Agent(llm=llm, browser=browser)

Environment variables:
    BROWSER_USE_SETUP_LOGGING: Control logging setup (default: 'true')
    BROWSER_USE_DEBUG_LOG_FILE: Path for debug log file (optional)
    BROWSER_USE_INFO_LOG_FILE: Path for info log file (optional)
    BROWSER_USE_LOGGING_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR)
    OPENAI_API_KEY: API key for OpenAI models
    ANTHROPIC_API_KEY: API key for Anthropic Claude models
    GOOGLE_API_KEY: API key for Google models
    GROQ_API_KEY: API key for Groq models
    ANONYMIZED_TELEMETRY: Enable/disable telemetry (default: 'true', set to 'false' to disable)
    BROWSER_USE_CLOUD_SYNC: Enable/disable cloud sync (default: same as ANONYMIZED_TELEMETRY)
    BROWSER_USE_CLOUD_API_URL: Cloud API endpoint (default: 'https://api.browser-use.com')
    BROWSER_USE_CLOUD_UI_URL: Cloud UI URL (set to empty string to disable)
    BROWSER_USE_CACHE_DIR: Directory for browser cache and profiles

Privacy and Telemetry:
    By default, browser-use collects anonymized telemetry data to improve the library.
    This includes usage statistics but no personal or sensitive information.
    
    To completely disable all telemetry and cloud features:
        >>> import os
        >>> # Disable before importing browser_use
        >>> os.environ['ANONYMIZED_TELEMETRY'] = 'false'
        >>> os.environ['BROWSER_USE_CLOUD_SYNC'] = 'false'
        >>> os.environ['BROWSER_USE_CLOUD_API_URL'] = 'http://localhost:9999'
        >>> os.environ['BROWSER_USE_CLOUD_UI_URL'] = ''
        >>> from browser_use import Agent
    
    Or set in your .env file:
        ANONYMIZED_TELEMETRY=false
        BROWSER_USE_CLOUD_SYNC=false
        BROWSER_USE_CLOUD_API_URL=http://localhost:9999
        BROWSER_USE_CLOUD_UI_URL=
    
    What gets disabled:
        - PostHog analytics (usage statistics)
        - Cloud session sync (browser-use.com integration)
        - Automatic session sharing features
        - All external network calls for telemetry
    
    The library works fully offline when telemetry is disabled.

Note:
    The library uses lazy imports for performance optimization. Components are 
    only imported when first accessed, reducing initial import time significantly.
    
    Logging is automatically configured unless BROWSER_USE_SETUP_LOGGING is set 
    to 'false' or when running in MCP mode.

Since: v1.0.0
"""
import os
from typing import TYPE_CHECKING

from browser_use.logging_config import setup_logging

# Only set up logging if not in MCP mode or if explicitly requested
if os.environ.get('BROWSER_USE_SETUP_LOGGING', 'true').lower() != 'false':
	from browser_use.config import CONFIG

	# Get log file paths from config/environment
	debug_log_file = getattr(CONFIG, 'BROWSER_USE_DEBUG_LOG_FILE', None)
	info_log_file = getattr(CONFIG, 'BROWSER_USE_INFO_LOG_FILE', None)

	# Set up logging with file handlers if specified
	logger = setup_logging(debug_log_file=debug_log_file, info_log_file=info_log_file)
else:
	import logging

	logger = logging.getLogger('browser_use')

# Monkeypatch BaseSubprocessTransport.__del__ to handle closed event loops gracefully
from asyncio import base_subprocess

_original_del = base_subprocess.BaseSubprocessTransport.__del__


def _patched_del(self):
	"""Patched __del__ that handles closed event loops gracefully.
	
	This internal patch prevents noisy RuntimeError exceptions when event loops 
	are closed during subprocess cleanup. The standard asyncio subprocess 
	transport cleanup can fail when the event loop is already closed, causing 
	unnecessary error messages.
	
	Note:
		This is an internal implementation detail that patches asyncio's 
		BaseSubprocessTransport to handle closed event loops without raising 
		RuntimeError exceptions.
	"""
	try:
		# Check if the event loop is closed before calling the original
		if hasattr(self, '_loop') and self._loop and self._loop.is_closed():
			# Event loop is closed, skip cleanup that requires the loop
			return
		_original_del(self)
	except RuntimeError as e:
		if 'Event loop is closed' in str(e):
			# Silently ignore this specific error
			pass
		else:
			raise


base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


# Type stubs for lazy imports - fixes linter warnings
if TYPE_CHECKING:
	from browser_use.agent.prompts import SystemPrompt
	from browser_use.agent.service import Agent
	from browser_use.agent.views import ActionModel, ActionResult, AgentHistoryList
	from browser_use.browser import BrowserProfile, BrowserSession
	from browser_use.browser import BrowserSession as Browser
	from browser_use.dom.service import DomService
	from browser_use.llm.anthropic.chat import ChatAnthropic
	from browser_use.llm.azure.chat import ChatAzureOpenAI
	from browser_use.llm.google.chat import ChatGoogle
	from browser_use.llm.groq.chat import ChatGroq
	from browser_use.llm.ollama.chat import ChatOllama
	from browser_use.llm.openai.chat import ChatOpenAI
	from browser_use.tools.service import Controller, Tools


# Lazy imports mapping - only import when actually accessed
_LAZY_IMPORTS = {
	# Agent service (heavy due to dependencies)
	'Agent': ('browser_use.agent.service', 'Agent'),
	# System prompt (moderate weight due to agent.views imports)
	'SystemPrompt': ('browser_use.agent.prompts', 'SystemPrompt'),
	# Agent views (very heavy - over 1 second!)
	'ActionModel': ('browser_use.agent.views', 'ActionModel'),
	'ActionResult': ('browser_use.agent.views', 'ActionResult'),
	'AgentHistoryList': ('browser_use.agent.views', 'AgentHistoryList'),
	'BrowserSession': ('browser_use.browser', 'BrowserSession'),
	'Browser': ('browser_use.browser', 'BrowserSession'),  # Alias for BrowserSession
	'BrowserProfile': ('browser_use.browser', 'BrowserProfile'),
	# Tools (moderate weight)
	'Tools': ('browser_use.tools.service', 'Tools'),
	'Controller': ('browser_use.tools.service', 'Controller'),  # alias
	# DOM service (moderate weight)
	'DomService': ('browser_use.dom.service', 'DomService'),
	# Chat models (very heavy imports)
	'ChatOpenAI': ('browser_use.llm.openai.chat', 'ChatOpenAI'),
	'ChatGoogle': ('browser_use.llm.google.chat', 'ChatGoogle'),
	'ChatAnthropic': ('browser_use.llm.anthropic.chat', 'ChatAnthropic'),
	'ChatGroq': ('browser_use.llm.groq.chat', 'ChatGroq'),
	'ChatAzureOpenAI': ('browser_use.llm.azure.chat', 'ChatAzureOpenAI'),
	'ChatOllama': ('browser_use.llm.ollama.chat', 'ChatOllama'),
}


def __getattr__(name: str):
	"""Lazy import mechanism for deferred module loading.
	
	Implements a lazy import system that only loads modules when they are first 
	accessed. This significantly reduces initial import time by deferring heavy 
	imports until they are actually needed.
	
	Args:
		name: The name of the attribute being accessed. Should be one of the 
			 names defined in _LAZY_IMPORTS.
	
	Returns:
		The imported attribute (class or function) from the specified module.
	
	Raises:
		ImportError: If the module or attribute cannot be imported.
		AttributeError: If the requested name is not in the lazy imports registry.
	
	Note:
		Once an attribute is imported, it's cached in the module's globals to 
		avoid repeated import overhead.
	"""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module

			module = import_module(module_path)
			attr = getattr(module, attr_name)
			# Cache the imported attribute in the module's globals
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	'Agent',
	'BrowserSession',
	'Browser',  # Alias for BrowserSession
	'BrowserProfile',
	'Controller',
	'DomService',
	'SystemPrompt',
	'ActionResult',
	'ActionModel',
	'AgentHistoryList',
	# Chat models
	'ChatOpenAI',
	'ChatGoogle',
	'ChatAnthropic',
	'ChatGroq',
	'ChatAzureOpenAI',
	'ChatOllama',
	'Tools',
	'Controller',
]
