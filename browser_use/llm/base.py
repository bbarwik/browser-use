"""Base protocol for language model implementations.

This module defines the protocol (interface) that all LLM implementations
must follow. The library has migrated from LangChain to OpenAI's message
format for better consistency and type safety.
"""

from typing import Any, Protocol, TypeVar, overload, runtime_checkable

from pydantic import BaseModel

from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion

T = TypeVar('T', bound=BaseModel)


@runtime_checkable
class BaseChatModel(Protocol):
	"""Protocol defining the interface for all language model implementations.
	
	@public
	
	All LLM implementations (OpenAI, Anthropic, Google, etc.) must conform to
	this protocol. This ensures consistent behavior across different providers
	and enables type-safe usage throughout the library.
	
	The protocol uses Python's Protocol typing (PEP 544) to define structural
	typing - any class that implements these methods and properties is considered
	a valid BaseChatModel, without explicit inheritance.
	
	Attributes:
		model: The model identifier (e.g., 'gpt-4', 'claude-3-opus').
		provider: The provider name (e.g., 'openai', 'anthropic').
		name: Human-readable name for the model.
		model_name: Legacy alias for model attribute.
	
	Example:
		>>> from browser_use.llm.openai import ChatOpenAI
		>>> from browser_use.llm.anthropic import ChatAnthropic
		>>> 
		>>> # Any of these implementations work with Agent
		>>> llm1 = ChatOpenAI(model='gpt-4')
		>>> llm2 = ChatAnthropic(model='claude-3-opus')
		>>> 
		>>> # Type checking ensures compatibility
		>>> def process(llm: BaseChatModel):
		...     return llm.model
		>>> 
		>>> process(llm1)  # Works
		>>> process(llm2)  # Also works
	
	Note:
		This is a Protocol class, not meant to be instantiated directly.
		Use one of the concrete implementations like ChatOpenAI, ChatAnthropic, etc.
	
	See Also:
		ChatOpenAI: OpenAI implementation
		ChatAnthropic: Anthropic/Claude implementation
		ChatGoogle: Google/Gemini implementation
	"""
	_verified_api_keys: bool = False

	model: str

	@property
	def provider(self) -> str: ...

	@property
	def name(self) -> str: ...

	@property
	def model_name(self) -> str:
		# for legacy support
		return self.model

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""Invoke the chat model asynchronously.
		
		Args:
			messages: List of messages to send to the model.
			output_format: Optional output format type for structured responses.
			
		Returns:
			Chat completion response with optional structured output.
		"""
		...

	@classmethod
	def __get_pydantic_core_schema__(
		cls,
		source_type: type,
		handler: Any,
	) -> Any:
		"""Allow this Protocol to be used in Pydantic models -> very useful to typesafe the agent settings for example.
		Returns a schema that allows any object (since this is a Protocol).
		"""
		from pydantic_core import core_schema

		# Return a schema that accepts any object for Protocol types
		return core_schema.any_schema()
