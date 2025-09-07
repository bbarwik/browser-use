"""OpenAI-like API implementations."""
from dataclasses import dataclass

from browser_use.llm.openai.chat import ChatOpenAI


@dataclass
class ChatOpenAILike(ChatOpenAI):
	"""Base class for OpenAI-compatible API providers.
	
	@public
	
	A class for interacting with any LLM provider that implements the OpenAI API 
	schema. This serves as a base class for providers that are compatible with 
	OpenAI's API format but run on different infrastructure (e.g., Azure OpenAI, 
	local Ollama servers, or other OpenAI-compatible services).
	
	This class extends ChatOpenAI and inherits all its functionality, allowing
	seamless integration with providers that follow the OpenAI API specification.
	
	Args:
		model: The name of the model to use.
		base_url: Optional custom API endpoint URL.
		api_key: API key for authentication (if required).
		
	Example:
		>>> # Use with a local OpenAI-compatible server
		>>> llm = ChatOpenAILike(
		...     model="llama-3",
		...     base_url="http://localhost:11434/v1"
		... )
		>>> 
		>>> # Use with Azure OpenAI (via ChatAzureOpenAI subclass)
		>>> from browser_use.llm.azure import ChatAzureOpenAI
		>>> azure_llm = ChatAzureOpenAI(
		...     model="gpt-4",
		...     deployment_name="my-deployment"
		... )
	"""

	model: str
