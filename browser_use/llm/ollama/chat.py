"""Ollama chat model implementation."""
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import httpx
from ollama import AsyncClient as OllamaAsyncClient
from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.ollama.serializer import OllamaMessageSerializer
from browser_use.llm.views import ChatInvokeCompletion

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatOllama(BaseChatModel):
	"""Ollama chat model integration for browser automation.
	
	@public
	
	Provides access to locally-hosted open-source models through Ollama,
	enabling fully private and offline browser automation. Supports various
	models including Llama 3, Mixtral, Mistral, Gemma, and custom fine-tuned models.
	
	Constructor Parameters:
		model: Model name available in your Ollama installation
			Common models:
			- "llama3.2": Latest Llama model (recommended for general use)
			- "llama3.1:70b": Larger Llama for complex tasks
			- "mixtral": MoE model with good performance
			- "mistral": Fast and efficient model
			- "gemma2": Google's lightweight model
			- "qwen2.5": Strong multilingual support
			- "deepseek-coder-v2": Specialized for code tasks
			Run `ollama list` to see available models on your system
			
		host: Ollama server URL (default: "http://localhost:11434")
			For remote Ollama: "http://your-server:11434"
			
		timeout: Request timeout in seconds (default: None)
		
		client_params: Additional parameters for Ollama client
			Example: {"headers": {"Authorization": "Bearer token"}}
	
	Installation:
		1. Install Ollama: https://ollama.com/download
		2. Pull a model: `ollama pull llama3.2`
		3. Ollama server starts automatically on port 11434
	
	Feature Support:
		- Structured Output: Yes (via JSON mode)
		- Tool Calling: Model-dependent
		- Vision: Yes (with vision models like llava, bakllava)
		- Context: Varies by model (4k-128k tokens)
		- Privacy: 100% local, no external API calls
		
	Vision Models:
		For multimodal tasks, use vision-capable models:
		- "llava": General vision understanding
		- "bakllava": Enhanced vision capabilities
		- "moondream": Lightweight vision model
		Note: Set use_vision=True in Agent when using vision models
	
	Example:
		>>> # Basic usage with default local Ollama
		>>> llm = ChatOllama(model="llama3.2")
		>>> agent = Agent(task="Navigate local site", llm=llm)
		
		>>> # Using remote Ollama server
		>>> llm = ChatOllama(
		...     model="mixtral",
		...     host="http://192.168.1.100:11434",
		...     timeout=60
		... )
		
		>>> # With vision model
		>>> llm = ChatOllama(model="llava")
		>>> agent = Agent(task="Analyze images", llm=llm, use_vision=True)
		
		>>> # Custom fine-tuned model
		>>> llm = ChatOllama(model="my-custom-model:latest")
	
	Performance Tips:
		- Smaller models (7B) are faster but less capable
		- Larger models (70B+) need significant RAM (32GB+)
		- Use quantized versions for better memory efficiency
		- GPU acceleration requires CUDA or Metal support
		
	Troubleshooting:
		- If connection fails, ensure Ollama is running: `ollama serve`
		- Check model is pulled: `ollama list`
		- For slow inference, try smaller or quantized models
		- Set OLLAMA_NUM_PARALLEL for concurrent requests
	"""

	model: str

	# # Model params
	# TODO (matic): Why is this commented out?
	# temperature: float | None = None

	# Client initialization parameters
	host: str | None = None
	timeout: float | httpx.Timeout | None = None
	client_params: dict[str, Any] | None = None

	# Static
	@property
	def provider(self) -> str:
		return 'ollama'

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary."""
		return {
			'host': self.host,
			'timeout': self.timeout,
			'client_params': self.client_params,
		}

	def get_client(self) -> OllamaAsyncClient:
		"""Returns an OllamaAsyncClient client.
		"""
		return OllamaAsyncClient(host=self.host, timeout=self.timeout, **self.client_params or {})

	@property
	def name(self) -> str:
		return self.model

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""Asynchronously invoke the Ollama chat model."""
		ollama_messages = OllamaMessageSerializer.serialize_messages(messages)

		try:
			if output_format is None:
				response = await self.get_client().chat(
					model=self.model,
					messages=ollama_messages,
				)

				return ChatInvokeCompletion(completion=response.message.content or '', usage=None)
			else:
				schema = output_format.model_json_schema()

				response = await self.get_client().chat(
					model=self.model,
					messages=ollama_messages,
					format=schema,
				)

				completion = response.message.content or ''
				if output_format is not None:
					completion = output_format.model_validate_json(completion)

				return ChatInvokeCompletion(completion=completion, usage=None)

		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
