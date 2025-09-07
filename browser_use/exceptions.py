"""Browser-use custom exception definitions."""


class LLMException(Exception):
	"""Exception raised when LLM API calls fail or return invalid responses.
	
	@public
	
	LLMException is raised when communication with the Large Language Model
	fails due to network issues, authentication problems, rate limiting,
	or invalid responses. This exception provides structured error information
	including HTTP status codes and detailed error messages.
	
	Common Causes:
		- Invalid or missing API key (401 Unauthorized)
		- Rate limit exceeded (429 Too Many Requests)
		- Insufficient credits/quota (402 Payment Required)
		- Model overloaded or unavailable (503 Service Unavailable)
		- Invalid model name or parameters (400 Bad Request)
		- Network timeout or connection issues
		- Malformed response from LLM provider
	
	This exception is commonly raised during browser automation when the
	LLM cannot process browser state or generate valid actions due to
	service disruptions or malformed responses.
	
	Attributes:
		status_code: HTTP status code from the LLM API response.
		message: Detailed error message describing the failure.
	
	Example:
		>>> try:
		...     result = llm_client.generate_action(browser_state)
		... except LLMException as e:
		...     if e.status_code == 429:
		...         print(f"Rate limited: {e.message}")
		...     elif e.status_code >= 500:
		...         print(f"Server error: {e.message}")
		...     else:
		...         print(f"Client error {e.status_code}: {e.message}")
	
	Note:
		Status codes follow HTTP conventions: 4xx for client errors
		(authentication, rate limits) and 5xx for server errors.
	"""
	def __init__(self, status_code, message):
		self.status_code = status_code
		self.message = message
		super().__init__(f'Error {status_code}: {message}')
