"""OpenRouter chat model implementation."""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.shared_params.response_format_json_schema import (
	JSONSchema,
	ResponseFormatJSONSchema,
)
from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.openrouter.serializer import OpenRouterMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatOpenRouter(BaseChatModel):
	"""OpenRouter chat model integration for browser automation.
	
	@public
	
	Provides access to 100+ language models through a unified API gateway,
	including GPT-4, Claude, Gemini, Llama, and many specialized models.
	OpenRouter automatically routes requests to available providers and
	handles failover for maximum reliability.
	
	Constructor Parameters:
		model: Model identifier from OpenRouter's catalog
			Popular models:
			- "openai/gpt-4-turbo": GPT-4 Turbo
			- "anthropic/claude-3.5-sonnet": Claude 3.5 Sonnet
			- "google/gemini-pro": Gemini Pro
			- "meta-llama/llama-3-70b": Llama 3 70B
			- "mistralai/mistral-large": Mistral Large
			See https://openrouter.ai/models for full list
			
		api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
			Get your key at https://openrouter.ai/keys
			
		temperature: Sampling temperature 0-2 (model-dependent)
		
		top_p: Nucleus sampling threshold
		
		seed: Random seed for deterministic outputs
		
		http_referer: Your app URL for OpenRouter analytics (optional)
			Example: "https://myapp.com"
			
		base_url: API endpoint (default: "https://openrouter.ai/api/v1")
		
		timeout: Request timeout in seconds
		
		max_retries: Retry attempts on failure (default: 10)
	
	Model Selection:
		OpenRouter supports model routing with fallbacks:
		- Use specific model: "anthropic/claude-3.5-sonnet"
		- Use category: "anthropic/claude-3.5-sonnet|openai/gpt-4" (fallback)
		- Check pricing at https://openrouter.ai/models
	
	Feature Support:
		Feature support varies by model. Common capabilities:
		- Tool Calling: GPT-4, Claude, some open models
		- Structured Output: GPT-4, Claude via JSON mode
		- Vision: GPT-4V, Claude 3, Gemini Pro Vision
		- Context: Varies widely (4k-200k tokens)
		
	Pricing:
		OpenRouter uses credit-based pricing. Each model has different
		costs per token. Set up billing at https://openrouter.ai/credits
		
	Benefits:
		- Single API for multiple providers
		- Automatic failover and load balancing
		- No need for multiple API keys
		- Unified billing across providers
		- Access to rare and specialized models
	
	Example:
		>>> # Using GPT-4 through OpenRouter
		>>> llm = ChatOpenRouter(
		...     model="openai/gpt-4-turbo",
		...     api_key="sk-or-...",  # or set OPENROUTER_API_KEY
		...     temperature=0.7,
		...     http_referer="https://myapp.com",
		... )
		>>> agent = Agent(task="Complex analysis", llm=llm)
		
		>>> # Using Claude with fallback to GPT-4
		>>> llm = ChatOpenRouter(
		...     model="anthropic/claude-3.5-sonnet|openai/gpt-4-turbo",
		...     api_key="sk-or-...",
		... )
		
		>>> # Using open-source model
		>>> llm = ChatOpenRouter(
		...     model="meta-llama/llama-3-70b",
		...     temperature=0.5,
		... )
	
	Rate Limits:
		OpenRouter enforces rate limits based on your account tier.
		The library automatically handles rate limit errors with
		exponential backoff. Upgrade at https://openrouter.ai/credits
		for higher limits.
	"""

	# Model configuration
	model: str

	# Model params
	temperature: float | None = None
	top_p: float | None = None
	seed: int | None = None

	# Client initialization parameters
	api_key: str | None = None
	http_referer: str | None = None  # OpenRouter specific parameter for tracking
	base_url: str | httpx.URL = 'https://openrouter.ai/api/v1'
	timeout: float | httpx.Timeout | None = None
	max_retries: int = 10
	default_headers: Mapping[str, str] | None = None
	default_query: Mapping[str, object] | None = None
	http_client: httpx.AsyncClient | None = None
	_strict_response_validation: bool = False

	# Static
	@property
	def provider(self) -> str:
		return 'openrouter'

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary."""
		# Define base client params
		base_params = {
			'api_key': self.api_key,
			'base_url': self.base_url,
			'timeout': self.timeout,
			'max_retries': self.max_retries,
			'default_headers': self.default_headers,
			'default_query': self.default_query,
			'_strict_response_validation': self._strict_response_validation,
			'top_p': self.top_p,
			'seed': self.seed,
		}

		# Create client_params dict with non-None values
		client_params = {k: v for k, v in base_params.items() if v is not None}

		# Add http_client if provided
		if self.http_client is not None:
			client_params['http_client'] = self.http_client

		return client_params

	def get_client(self) -> AsyncOpenAI:
		"""Returns an AsyncOpenAI client configured for OpenRouter.

		Returns:
		    AsyncOpenAI: An instance of the AsyncOpenAI client with OpenRouter base URL.
		"""
		if not hasattr(self, '_client'):
			client_params = self._get_client_params()
			self._client = AsyncOpenAI(**client_params)
		return self._client

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
		"""Extract usage information from the OpenRouter response."""
		if response.usage is None:
			return None

		prompt_details = getattr(response.usage, 'prompt_tokens_details', None)
		cached_tokens = prompt_details.cached_tokens if prompt_details else None

		return ChatInvokeUsage(
			prompt_tokens=response.usage.prompt_tokens,
			prompt_cached_tokens=cached_tokens,
			prompt_cache_creation_tokens=None,
			prompt_image_tokens=None,
			# Completion
			completion_tokens=response.usage.completion_tokens,
			total_tokens=response.usage.total_tokens,
		)

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""Invoke the model with the given messages through OpenRouter.

		Args:
		    messages: List of chat messages
		    output_format: Optional Pydantic model class for structured output

		Returns:
		    Either a string response or an instance of output_format
		"""
		openrouter_messages = OpenRouterMessageSerializer.serialize_messages(messages)

		# Set up extra headers for OpenRouter
		extra_headers = {}
		if self.http_referer:
			extra_headers['HTTP-Referer'] = self.http_referer

		try:
			if output_format is None:
				# Return string response
				response = await self.get_client().chat.completions.create(
					model=self.model,
					messages=openrouter_messages,
					temperature=self.temperature,
					top_p=self.top_p,
					seed=self.seed,
					extra_headers=extra_headers,
				)

				usage = self._get_usage(response)
				return ChatInvokeCompletion(
					completion=response.choices[0].message.content or '',
					usage=usage,
				)

			else:
				# Create a JSON schema for structured output
				schema = SchemaOptimizer.create_optimized_json_schema(output_format)

				response_format_schema: JSONSchema = {
					'name': 'agent_output',
					'strict': True,
					'schema': schema,
				}

				# Return structured response
				response = await self.get_client().chat.completions.create(
					model=self.model,
					messages=openrouter_messages,
					temperature=self.temperature,
					top_p=self.top_p,
					seed=self.seed,
					response_format=ResponseFormatJSONSchema(
						json_schema=response_format_schema,
						type='json_schema',
					),
					extra_headers=extra_headers,
				)

				if response.choices[0].message.content is None:
					raise ModelProviderError(
						message='Failed to parse structured output from model response',
						status_code=500,
						model=self.name,
					)
				usage = self._get_usage(response)

				parsed = output_format.model_validate_json(response.choices[0].message.content)

				return ChatInvokeCompletion(
					completion=parsed,
					usage=usage,
				)

		except RateLimitError as e:
			raise ModelRateLimitError(message=e.message, model=self.name) from e

		except APIConnectionError as e:
			raise ModelProviderError(message=str(e), model=self.name) from e

		except APIStatusError as e:
			raise ModelProviderError(message=e.message, status_code=e.status_code, model=self.name) from e

		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
