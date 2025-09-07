# browser-use API Reference

## Navigation Guide

**For Humans:**
- Use `grep -n '^##' API.md` to list all main sections with line numbers
- Use `grep -n '^###' API.md` to list all classes and functions
- Use `grep -n '^####' API.md` to list all methods and properties
- Search for specific features: `grep -n -i "ClassName" API.md` or `grep -n -i "function_name" API.md`

**For AI Assistants:**
- Use the Grep tool with pattern `^##` to list all module sections (e.g., `^## browser_use.module`)
- Use pattern `^###` to find all classes and functions (e.g., `### ClassName`, `### function_name`)
- Use pattern `^####` to find all methods (e.g., `#### ClassName.method_name`)
- For specific lookups, use patterns like `class AIMessages` or `def generate` with output_mode="content" and -n=true for line numbers
- Use -C flag (context lines) to see surrounding content: `pattern="AIMessages", -C=5`
- Navigate directly to line numbers using Read tool with offset parameter once you know the location


## browser_use

Browser automation library with AI capabilities using LLMs and Playwright.

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

**Example**:

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
- `BROWSER_USE_SETUP_LOGGING` - Control logging setup (default: 'true')
- `BROWSER_USE_DEBUG_LOG_FILE` - Path for debug log file (optional)
- `BROWSER_USE_INFO_LOG_FILE` - Path for info log file (optional)
- `BROWSER_USE_LOGGING_LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR)
- `OPENAI_API_KEY` - API key for OpenAI models
- `ANTHROPIC_API_KEY` - API key for Anthropic Claude models
- `GOOGLE_API_KEY` - API key for Google models
- `GROQ_API_KEY` - API key for Groq models
- `ANONYMIZED_TELEMETRY` - Enable/disable telemetry (default: 'true', set to 'false' to disable)
- `BROWSER_USE_CLOUD_SYNC` - Enable/disable cloud sync (default: same as ANONYMIZED_TELEMETRY)
- `BROWSER_USE_CLOUD_API_URL` - Cloud API endpoint (default: 'https://api.browser-use.com')
- `BROWSER_USE_CLOUD_UI_URL` - Cloud UI URL (set to empty string to disable)
- `BROWSER_USE_CACHE_DIR` - Directory for browser cache and profiles

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

**Notes**:

  The library uses lazy imports for performance optimization. Components are
  only imported when first accessed, reducing initial import time significantly.

  Logging is automatically configured unless BROWSER_USE_SETUP_LOGGING is set
  to 'false' or when running in MCP mode.

- `Since` - v1.0.0


## browser_use.agent.prompts


## browser_use.agent.views

### ActionResult

```python
class ActionResult(BaseModel)
```

Result of executing an action.

ActionResult encapsulates the outcome of any action performed by the agent.
It can indicate completion, errors, extracted content, and memory updates.
Custom actions should return an ActionResult instance.

Fields:
is_done (optional): Set to True to mark task as complete (default: False).
success (optional): True/False to indicate task success when done.
error (optional): Error message if action failed, added to long-term memory.
attachments (optional): List of file paths to display in completion message.
long_term_memory (optional): Persistent memory update for this action.
extracted_content (optional): Data extracted from the page or action.
include_extracted_content_only_once (optional): If True, extracted_content
is only used for the next step (default: False).
metadata (optional): Additional data for observability (e.g., click coords).
include_in_memory (deprecated): Use long_term_memory instead.

How the agent interprets results:
- is_done=True: Stops execution and completes the task
- success=True: Marks task as successfully completed (requires is_done=True)
- error: Logs error, increments failure count, may trigger retry
- extracted_content: Added to agent's context for decision making
- long_term_memory: Persisted across all future steps
- include_extracted_content_only_once=True: Content only visible to next step
- attachments: File paths shown to user in completion message

Memory Trade-offs:
long_term_memory: Use for concise, important facts (e.g., "User ID: 12345")
- Always included in context
- Should be brief to avoid token bloat
- Persists across all steps

extracted_content: Use for large data to process (e.g., full article text)
- Can be lengthy without permanent impact
- With include_extracted_content_only_once=True: only for next step
- Good for temporary processing of large content

**Example**:

  >>> @tools.registry.action("Extract price")
  ... async def extract_price(params, browser_session):
  ...     # ... extraction logic ...
  ...     return ActionResult(
  ...         extracted_content="Price: $99.99",
  ...         include_extracted_content_only_once=True
  ...     )
  >>>
  >>> # Mark task complete
  >>> return ActionResult(
  ...     is_done=True,
  ...     success=True,
  ...     extracted_content="Task completed successfully"
  ... )
  >>>
  >>> # Report error
  >>> return ActionResult(
  ...     error="Failed to find element",
  ...     long_term_memory="Could not locate price on page"
  ... )

### AgentOutput

```python
class AgentOutput(BaseModel)
```

Complete output from the agent's LLM call.

Contains the agent's cognitive state and planned actions. This is the
primary response format from the LLM, including both the agent's
internal reasoning and the actions it wants to execute.

**Attributes**:

- `thinking` - Optional reasoning about the current situation.
- `evaluation_previous_goal` - Assessment of previous action's success.
- `memory` - Important information to remember.
- `next_goal` - Plan for the next action.
- `action` - List of actions to execute (required, min 1).
- `current_state` - Alternative AgentBrain object (XOR with individual fields).
- `result` - Extracted structured data (if output schema provided).

**Example**:

  >>> output = AgentOutput(
  ...     thinking="Need to click the login button",
  ...     evaluation_previous_goal="Successfully navigated to login page",
  ...     memory="Username field has id 'user-input'",
  ...     next_goal="Enter credentials",
  ...     action=[ActionModel(...)]
  ... )

**Notes**:

  Either provide individual fields (thinking, evaluation_previous_goal, etc.)
  OR provide current_state, but not both.

### AgentHistory

```python
class AgentHistory(BaseModel)
```

History item for agent actions.

Represents a single step in the agent's execution history, containing
the LLM output, action results, browser state, and timing metadata.

**Attributes**:

- `model_output` - The AgentOutput from the LLM for this step, containing
  the agent's reasoning and planned actions.
- `result` - List of ActionResult objects from executing the actions.
- `state` - BrowserStateHistory snapshot at this step.
- `metadata` - StepMetadata with timing information.

**Example**:

  >>> history_item = agent_history_list.history[0]
  >>> print(f"Step {history_item.metadata.step_number}")
  >>> print(f"Goal: {history_item.model_output.next_goal}")
  >>> print(f"Actions taken: {len(history_item.result)}")
  >>> print(f"URL: {history_item.state.url}")

### AgentHistoryList

```python
class AgentHistoryList(BaseModel, Generic[AgentStructuredOutput])
```

List of AgentHistory messages, i.e. the history of the agent's actions and thoughts.

The object returned by agent.run(), used to access the history of actions,
results, errors, and the final extracted content from the agent's execution.

Fields:
history: List of AgentHistory items, one per execution step.
usage: Token usage statistics including costs (when calculate_cost=True).
_output_model_schema: Internal reference to structured output schema.

Properties:
structured_output: Parsed structured output if output_model_schema was provided.

Helper Methods:
final_result(): Get the final extracted content string.
errors(): Get list of errors (None for successful steps).
urls(): Get list of visited URLs.
screenshot_paths(): Get paths to saved screenshots.
screenshots(): Get screenshots as base64 strings.
action_names(): Get list of action names executed.
model_thoughts(): Get agent's reasoning at each step.
model_outputs(): Get raw LLM outputs.
model_actions(): Get all actions with parameters.
action_results(): Get all ActionResult objects.
extracted_content(): Get all extracted content strings.
is_done(): Check if agent marked task as complete.
is_successful(): Check if agent deemed task successful.
has_errors(): Check if any errors occurred.
total_duration_seconds(): Get total execution time.
number_of_steps(): Get step count.

Serialization:
save_to_file(filepath): Save history to JSON file.
model_dump(): Convert to dictionary for serialization.
Can be restored with AgentHistoryList.model_validate(data).

**Example**:

  >>> history = await agent.run()
  >>> print(f"Steps: {history.number_of_steps()}")
  >>> print(f"Result: {history.final_result()}")
  >>> if history.has_errors():
  ...     for error in history.errors():
  ...         if error: print(f"Error: {error}")
  >>> history.save_to_file("agent_history.json")

#### AgentHistoryList.errors

```python
def errors(self) -> list[str | None]
```

Get all errors from history, with None for steps without errors.

Returns a list of errors that occurred during the agent's execution.
Each list item corresponds to a step, with None if no error occurred.

**Returns**:

  List of error strings or None for each step.

**Example**:

  >>> errors = history.errors()
  >>> for i, error in enumerate(errors):
  ...     if error:
  ...         print(f"Step {i} failed: {error}")

#### AgentHistoryList.final_result

```python
def final_result(self) -> None | str
```

Final result from history.

Retrieves the extracted content from the final step of the agent's execution.
This is typically the main output or answer the agent was tasked to find.

**Returns**:

  The extracted content string, or None if no content was extracted.

**Example**:

  >>> history = agent.run()
  >>> result = history.final_result()
  >>> if result:
  ...     print(f"Agent found: {result}")

#### AgentHistoryList.is_done

```python
def is_done(self) -> bool
```

Check if the agent is done.

A method to check if the agent's task has been completed. Returns True
when the agent has executed a 'done' action.

**Returns**:

  True if the agent marked the task as complete, False otherwise.

**Example**:

  >>> if history.is_done():
  ...     print("Task completed successfully")

#### AgentHistoryList.is_successful

```python
def is_successful(self) -> bool | None
```

Check if the agent completed successfully.

Returns the success status as determined by the agent itself in the final step.
This is only meaningful when the task is marked as done (is_done() returns True).

**Returns**:

  True if the agent marked the task as successfully completed.
  False if the agent marked the task as completed but failed.
  None if the task is not yet done or success status wasn't specified.

**Example**:

  >>> history = await agent.run()
  >>> if history.is_done():
  ...     if history.is_successful():
  ...         print("Task completed successfully")
  ...     elif history.is_successful() is False:
  ...         print("Task completed but failed")
  ...     else:
  ...         print("Task completed, success status unknown")

#### AgentHistoryList.urls

```python
def urls(self) -> list[str | None]
```

Get all URLs visited during agent execution.

Returns a list of URLs that the agent visited during its execution,
one per step. URLs may be None for steps where no page was loaded.

**Returns**:

  List of URL strings or None values, one per execution step.

**Example**:

  >>> history = await agent.run()
  >>> visited_urls = history.urls()
  >>> unique_urls = [url for url in visited_urls if url is not None]
  >>> print(f"Agent visited {len(set(unique_urls))} unique pages")

#### AgentHistoryList.model_thoughts

```python
def model_thoughts(self) -> list[AgentBrain]
```

Get all thoughts from history.

Provides the model's reasoning or "thoughts" at each step, including
evaluations of previous actions and plans for next steps.

**Returns**:

  List of AgentBrain objects containing the agent's cognitive state.

**Example**:

  >>> thoughts = history.model_thoughts()
  >>> for thought in thoughts:
  ...     print(f"Goal: {thought.next_goal}")

#### AgentHistoryList.model_actions

```python
def model_actions(self) -> list[dict]
```

Get all actions from history.

Returns the sequence of actions the model decided to take, including
their parameters and the elements they interacted with.

**Returns**:

  List of dictionaries containing action details and parameters.

**Example**:

  >>> actions = history.model_actions()
  >>> for action in actions:
  ...     print(f"Action: {action['name']}")


## browser_use.agent.message_manager.utils


## browser_use.agent.message_manager.views


## browser_use.agent.message_manager.service


## browser_use.agent.gif


## browser_use.agent.service

### Agent

```python
class Agent(Generic[Context, AgentStructuredOutput])
```

Main AI agent for browser automation tasks.

The Agent class is the primary interface for browser automation using LLMs.
It coordinates between the browser session, LLM, and tools to execute complex
web automation tasks. The agent maintains state across interactions, manages
conversation history, and handles retries on failures.

The agent uses a step-based execution model where each step involves:
1. Capturing the current browser state (DOM, screenshot, URL)
2. Building context with history, memory, and current state
3. Sending context to the LLM for decision making
4. Validating and executing the LLM's chosen actions
5. Processing action results (errors, extracted content, completion)
6. Updating internal state, memory, and history
7. Checking stop conditions (is_done, max_steps, failures)

Retry Mechanism:
The agent automatically retries on failures up to max_failures times
(default: 3) with exponential backoff. Rate limit errors trigger
automatic backoff. Consecutive failures increment a counter that
resets on successful actions.

Type Parameters:
Context: Type of custom context data passed to tools
AgentStructuredOutput: Type of structured output when using output schemas

Constructor Args:
task: Description of the task to perform. Should be clear and specific.
llm: Language model for decision making. Defaults to GPT-4 mini.
browser_profile: Browser profile configuration. If None, uses default profile.
browser_session: Existing browser session to use. Creates new if None.
browser: Alias for browser_session (preferred parameter name).
tools: Tool registry with custom actions. Creates default if None.
controller: Alias for tools parameter.
sensitive_data: Credentials to use during automation. Can be:
- Simple dict: {"username": "user", "password": "pass"}
- Domain-specific: {"github.com": {"username": "..."}}
initial_actions: List of actions to execute before starting the task.
output_model_schema: Pydantic model for structured output extraction.
use_vision: Whether to include screenshots in LLM context (default: True).
save_conversation_path: Path to save conversation history.
max_failures: Maximum retry attempts on step failure (default: 3).
override_system_message: Replace default system prompt entirely.
extend_system_message: Append to default system prompt.
generate_gif: Generate animated GIF of browser actions.
available_file_paths: List of file paths agent can access.
include_attributes: HTML attributes to include in DOM (default: minimal).
max_actions_per_step: Maximum actions per LLM call (default: 10).
use_thinking: Enable thinking/reasoning in prompts (default: True).
flash_mode: Use faster, less thorough mode (default: False).
max_history_items: Limit conversation history size.
page_extraction_llm: Separate LLM for page content extraction.
calculate_cost: Track token usage costs (default: False).
vision_detail_level: Image quality for vision ('auto', 'low', 'high').
llm_timeout: Timeout for LLM calls in seconds (default: 90).
step_timeout: Timeout for step execution in seconds (default: 120).
directly_open_url: Auto-navigate to URLs in task (default: True).
include_recent_events: Include recent browser events in context (default: False).

**Attributes**:

- `id` - Unique identifier for this agent instance
- `task_id` - Alias for id, used for task tracking
- `session_id` - Unique session identifier
- `task` - The task description provided to the agent
- `llm` - The language model used for decision making
- `browser_session` - The browser session for web interaction
- `tools` - Tool registry with available actions
- `state` - Current agent state including memory and goals
- `history` - Complete history of agent actions and responses
- `settings` - Configuration settings for the agent

  Internal State Components (agent.state):
- `memory` - Accumulated knowledge from the task execution
- `current_goal` - What the agent is trying to achieve now
- `last_result` - Results from the most recent action
- `n_steps` - Number of steps executed so far
- `consecutive_failures` - Failed attempts counter (resets on success)
- `paused/stopped` - Execution control flags
- `message_manager_state` - Conversation history and context
- `file_system_state` - Active file operations and paths

**Example**:

  >>> from browser_use import Agent, BrowserSession
  >>> from browser_use.llm.openai import ChatOpenAI
  >>>
  >>> async def automate_search():
  ...     browser = BrowserSession()
  ...     llm = ChatOpenAI(model="gpt-4")
  ...     agent = Agent(
  ...         task="Search for Python documentation",
  ...         llm=llm,
  ...         browser=browser
  ...     )
  ...     result = await agent.run()
  ...     return result

  >>> # With structured output
  >>> from pydantic import BaseModel
  >>> class SearchResult(BaseModel):
  ...     title: str
  ...     url: str
  ...     summary: str
  >>>
  >>> agent = Agent(
  ...     task="Extract search results",
  ...     output_model_schema=SearchResult
  ... )
  >>> history = await agent.run()
  >>>
  >>> # Access structured output
  >>> if history.structured_output:
  ...     result: SearchResult = history.structured_output
  ...     print(f"Title: {result.title}")
  ...     print(f"URL: {result.url}")

  Security:
- `IMPORTANT` - Always follow these security best practices:
  - Use allowed_domains to restrict where the agent can navigate
  - Store sensitive_data only for domains you trust
  - Enable headless=False during development to monitor agent actions
  - Never expose API keys or credentials in logs or screenshots
  - Use domain-specific credentials: {"github.com": {"token": "..."}}
  - Consider network isolation for sensitive automation tasks
  - Regularly audit agent history and extracted content

**Notes**:

  The agent requires an async context and should be used with
  `async with` or proper cleanup via the `close()` method.

**See Also**:

- `BrowserSession` - Browser control and session management
- `Tools` - Tool registry for available actions
- `AgentHistory` - History tracking for agent actions

#### Agent.add_new_task

```python
def add_new_task(self, new_task: str) -> None
```

Add a follow-up task for the agent to execute.

Queues a new task to be executed after the current task completes.
Useful for chaining multiple tasks or adding tasks dynamically.

**Arguments**:

- `new_task` - Description of the follow-up task to execute.

**Example**:

  >>> agent = Agent(task="Login to website")
  >>> agent.add_new_task("Download the report")
  >>> agent.add_new_task("Logout")
  >>>
  >>> # Tasks are processed sequentially
  >>> # Each call to add_new_task extends the current session
  >>> result = await agent.run(max_steps=50)
  >>>
  >>> # Or chain tasks manually
  >>> result1 = await agent.run()  # Executes "Login to website"
  >>> agent.add_new_task("Download report")
  >>> result2 = await agent.run()  # Continues with "Download report"

**Notes**:

  run() executes ONE task at a time and returns. To process multiple
  queued tasks, use a loop. The agent maintains context (cookies,
  session state) between tasks.

#### Agent.run

```python
@observe(name='agent.run', metadata={'task': '{{task}}', 'debug': '{{debug}}'})
@time_execution_async('--run')
async def run(self, max_steps: int = 100, on_step_start: AgentHookFunc | None = None, on_step_end: AgentHookFunc | None = None) -> AgentHistoryList[AgentStructuredOutput]
```

Execute the agent's task until completion or maximum steps.

Main execution method that runs the agent through multiple steps to
complete the assigned task. Each step involves capturing browser state,
deciding on actions, executing them, and updating history.

The agent will continue until:
- The task is marked as complete (done action)
- Maximum steps are reached
- An unrecoverable error occurs
- The user interrupts execution (Ctrl+C)

Browser Lifecycle:
After run() completes, the browser session is automatically closed
UNLESS browser_profile.keep_alive=True is set. The cleanup happens
in the finally block via self.close():

- If keep_alive=False/None (default): Browser is killed
- If keep_alive=True: Browser remains active for manual use

To keep browser alive for post-run interactions:
>>> profile = BrowserProfile(keep_alive=True)
>>> agent = Agent(browser=Browser(profile))
>>> await agent.run()
>>> # Browser is still alive here
>>> await agent.browser_session.get_browser_state_summary()
>>> await agent.browser_session.stop()  # Manual cleanup

**Arguments**:

- `max_steps` - Maximum number of steps to execute (default: 100).
  Prevents infinite loops.
- `on_step_start` - Optional async callback before each step.
- `on_step_end` - Optional async callback after each step.

**Returns**:

  AgentHistoryList containing:
  - history: List of all steps taken
  - usage: Token usage statistics (if calculate_cost=True)
  - result: Structured output (if output_model_schema provided)

**Raises**:

- `Exception` - Any unhandled exceptions during execution.

**Example**:

  >>> async def main():
  ...     agent = Agent(task="Search for Python docs")
  ...     result = await agent.run(max_steps=50)
  ...     print(f"Completed in {len(result.history)} steps")
  ...     if result.result:
  ...         print(f"Extracted data: {result.result}")

**Notes**:

  The agent automatically handles:
  - Browser lifecycle management (see Browser Lifecycle above)
  - Error recovery and retries
  - History tracking and GIF generation
  - Conversation saving (if configured)
  - Telemetry and cloud sync (if enabled)

**Warnings**:

  Always set appropriate max_steps to prevent runaway execution.
  Use Ctrl+C to pause/resume or double Ctrl+C to force exit.

#### Agent.load_and_rerun

```python
async def load_and_rerun(self, history_file: str | Path | None = None, **kwargs) -> list[ActionResult]
```

Load history from file and rerun it.

Loads a previously saved agent history from a JSON file and reruns the
actions. Useful for debugging, testing, or reproducing previous runs.

**Arguments**:

- `history_file` - Path to the history file (default: "AgentHistory.json")
- `**kwargs` - Additional arguments passed to rerun_history:
  - skip_failures: Whether to continue on action failures
  - delay_between_actions: Seconds to wait between actions

**Returns**:

  List of ActionResult objects from the rerun

  File Format:
  The history file should be a JSON file created by save_history(),
  containing the full agent history with actions, thoughts, and results.

  Limitations:
  - Browser state must match original conditions for accurate replay
  - Dynamic content (timestamps, random IDs) may cause differences
  - Authentication state must be restored separately

**Example**:

  >>> # Rerun a previous session
  >>> results = await agent.load_and_rerun("session_20240115.json")
  >>>
  >>> # Rerun with modifications
  >>> results = await agent.load_and_rerun(
  ...     "debug_session.json",
  ...     skip_failures=True,
  ...     delay_between_actions=1.0
  ... )

#### Agent.save_history

```python
def save_history(self, file_path: str | Path | None = None) -> None
```

Save agent history to a JSON file.

Saves the complete agent history including all steps, actions,
and responses to a JSON file for later analysis or replay.

**Arguments**:

- `file_path` - Path to save history. If None, uses default location
  in agent directory with timestamp.

**Example**:

  >>> agent.save_history("./logs/agent_history.json")
  >>> # Or use default location
  >>> agent.save_history()

**See Also**:

- `load_and_rerun` - Load and replay saved history

#### Agent.pause

```python
def pause(self) -> None
```

Pause agent execution.

Pauses the agent's execution at the next safe point. The agent
will complete any currently executing action before pausing.

**Example**:

  >>> agent.pause()
  >>> # Agent will pause after current action
  >>> agent.resume()  # Continue execution

**See Also**:

- `resume` - Resume paused execution
- `stop` - Permanently stop execution

#### Agent.resume

```python
def resume(self) -> None
```

Resume paused agent execution.

Resumes agent execution after being paused. Has no effect if
the agent is not currently paused.

**Example**:

  >>> agent.pause()
  >>> await asyncio.sleep(5)  # Wait 5 seconds
  >>> agent.resume()  # Continue execution

**See Also**:

- `pause` - Pause execution
- `stop` - Permanently stop execution

#### Agent.stop

```python
def stop(self) -> None
```

Stop agent execution permanently.

Sets a flag to stop the agent at the next safe point. Unlike pause(),
this permanently stops execution and cannot be resumed.

**Example**:

  >>> agent.stop()
  >>> # Agent will stop after current action
  >>> # Cannot be resumed

**See Also**:

- `pause` - Temporarily pause execution
- `resume` - Resume paused execution

#### Agent.close

```python
async def close(self)
```

Clean up agent resources and conditionally close browser session.

Properly closes agent resources and browser session based on keep_alive setting.
This method is automatically called at the end of Agent.run() in the finally block.

Browser Behavior:
- If browser_profile.keep_alive=False/None (default):
Browser session is killed, closing all tabs and the browser process
- If browser_profile.keep_alive=True:
Browser session remains active for manual operations
You must manually call browser_session.stop() or .kill() later

Resources Always Cleaned:
- File system handlers
- Telemetry and logs flushed
- Garbage collection triggered

This method should be called when done with the agent, either
explicitly or via async context manager.

**Example**:

  >>> # Default behavior - browser closes automatically
  >>> agent = Agent(task="...")
  >>> await agent.run()  # Browser killed in finally block

  >>> # Keep browser alive for manual operations
  >>> profile = BrowserProfile(keep_alive=True)
  >>> agent = Agent(task="...", browser=Browser(profile))
  >>> await agent.run()  # Browser stays alive
  >>> # Do manual operations...
  >>> await agent.browser_session.stop()  # Manual cleanup

  >>> # Using async context manager (always closes)
  >>> async with Agent(task="...") as agent:
  ...     await agent.run()

**Notes**:

  Always call close() to prevent resource leaks. If keep_alive=True,
  remember to manually stop the browser session when done.

#### Agent.run_sync

```python
def run_sync(self, max_steps: int = 100, on_step_start: AgentHookFunc | None = None, on_step_end: AgentHookFunc | None = None) -> AgentHistoryList[AgentStructuredOutput]
```

Synchronous wrapper around the async run method for easier usage without asyncio.

A synchronous version of run() for use in non-async contexts. Internally
uses asyncio.run() to execute the async run method.

**Arguments**:

- `max_steps` - Maximum number of steps to execute
- `on_step_start` - Optional callback before each step
- `on_step_end` - Optional callback after each step

**Returns**:

  AgentHistoryList containing the complete execution history

**Example**:

  >>> # Simple synchronous script
  >>> from browser_use import Agent, BrowserSession, ChatOpenAI
  >>>
  >>> # No async/await needed
  >>> browser = BrowserSession(headless=True)
  >>> llm = ChatOpenAI(model="gpt-4")
  >>> agent = Agent(
  ...     task="Find the latest Python version number",
  ...     llm=llm,
  ...     browser=browser
  ... )
  >>>
  >>> # Run synchronously - perfect for scripts and notebooks
  >>> history = agent.run_sync(max_steps=10)
  >>>
  >>> # Process results
  >>> if history.is_successful():
  ...     print(f"Result: {history.final_result()}")
  >>> else:
  ...     print(f"Failed: {history.errors()}")


## browser_use.agent.cloud_events

### UpdateAgentTaskEvent

```python
class UpdateAgentTaskEvent(BaseEvent)
```

Event for updating an agent task in the cloud.

Used to synchronize agent task state changes with cloud services including
completion status, pause/resume state, and output results. This enables
monitoring and control of agent tasks from external systems.

**Attributes**:

- `id` - Task ID to update
- `user_id` - User identifier for authorization
- `device_id` - Optional device ID for auth lookup
- `stopped` - Whether task has been stopped
- `paused` - Whether task is currently paused
- `done_output` - Final output when task completes
- `finished_at` - Timestamp of task completion
- `agent_state` - Current agent state dictionary
- `user_feedback_type` - Type of user feedback
- `user_comment` - User comment/feedback text
- `gif_url` - URL to generated GIF recording

**Example**:

  >>> event = UpdateAgentTaskEvent.from_agent(agent)
  >>> await cloud_sync.send_event(event)

### CreateAgentOutputFileEvent

```python
class CreateAgentOutputFileEvent(BaseEvent)
```

Event for creating agent output files in the cloud.

Used to upload files generated during agent execution like screenshots,
recordings, extracted content, or other artifacts to cloud storage.

**Attributes**:

- `id` - Unique file ID (auto-generated)
- `user_id` - User identifier for authorization
- `device_id` - Optional device ID for auth lookup
- `task_id` - Associated task ID
- `file_name` - Name of the file
- `file_content` - Base64 encoded file content
- `content_type` - MIME type of the file
- `created_at` - Timestamp of file creation

  File Size Limits:
  Maximum file size: 50MB (after base64 encoding)

**Example**:

  >>> # Upload a screenshot
  >>> with open('screenshot.png', 'rb') as f:
  ...     content = base64.b64encode(f.read()).decode()
  >>> event = CreateAgentOutputFileEvent(
  ...     task_id=agent.task_id,
  ...     file_name='screenshot.png',
  ...     file_content=content,
  ...     content_type='image/png'
  ... )


## browser_use.cli

Browser-Use command-line interface module.

This module provides both interactive textual and prompt-based CLI interfaces
for the browser-use library, allowing users to interact with AI agents that
can control web browsers.

CLI Modes:
TUI Mode (default): Interactive terminal UI with chat interface
- Run without arguments: `browser-use`
- Live browser control with visual feedback
- Command history and suggestions
- Keyboard shortcuts for navigation

One-Shot Mode: Execute single task and exit
- Use -p/--prompt flag: `browser-use -p "search for Python docs"`
- Automatically enables headless browser
- Returns results to stdout
- Good for scripting and automation

MCP Server Mode: Run as Model Context Protocol server
- Use --mcp flag: `browser-use --mcp`
- Exposes JSON-RPC via stdin/stdout
- Integrates with MCP-compatible clients
- Enables browser control from other applications

Command-Line Flags:
--version: Print version and exit
--model: LLM model name (gpt-4, claude-3-5-sonnet, gemini-2.0-flash)
--debug: Enable verbose startup logging
--headless: Run browser without UI (default for one-shot mode)
--window-width: Browser window width in pixels
--window-height: Browser window height in pixels
--user-data-dir: Chrome user data directory path
--profile-directory: Chrome profile name ("Default", "Profile 1")
--cdp-url: Connect to existing Chrome via CDP (http://localhost:9222)
--proxy-url: Proxy server URL (http://host:8080, socks5://host:1080)
--no-proxy: Comma-separated hosts to bypass proxy
--proxy-username: Proxy authentication username
--proxy-password: Proxy authentication password
-p/--prompt: Run single task without TUI
--mcp: Run as MCP server

Configuration:
Config File: ~/.config/browseruse/config.json
Command History: ~/.config/browseruse/command_history.json
Browser Profiles: ~/.config/browseruse/profiles/

Environment Variable Overrides:
BROWSER_USE_CONFIG_DIR: Change config directory location
BROWSER_USE_CONFIG_PATH: Use specific config file
BROWSER_USE_LOGGING_LEVEL: Set log verbosity

All LLM API keys are read from environment variables:
OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.

**Examples**:

  # Interactive TUI mode
  browser-use

  # One-shot task execution
  browser-use -p "Find the latest Python documentation"

  # Use specific model
  browser-use --model claude-3-5-sonnet

  # Connect to existing Chrome
  browser-use --cdp-url http://localhost:9222

  # Run with proxy
  browser-use --proxy-url http://proxy:8080 --proxy-username user

  # MCP server mode
  browser-use --mcp


## browser_use.exceptions

### LLMException

```python
class LLMException(Exception)
```

Exception raised when LLM API calls fail or return invalid responses.

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

**Attributes**:

- `status_code` - HTTP status code from the LLM API response.
- `message` - Detailed error message describing the failure.

**Example**:

  >>> try:
  ...     result = llm_client.generate_action(browser_state)
  ... except LLMException as e:
  ...     if e.status_code == 429:
  ...         print(f"Rate limited: {e.message}")
  ...     elif e.status_code >= 500:
  ...         print(f"Server error: {e.message}")
  ...     else:
  ...         print(f"Client error {e.status_code}: {e.message}")

**Notes**:

  Status codes follow HTTP conventions: 4xx for client errors
  (authentication, rate limits) and 5xx for server errors.


## browser_use.utils


## browser_use.tokens.views


## browser_use.tokens.tests.test_cost


## browser_use.tokens


## browser_use.tokens.service


## browser_use.llm.openai.serializer


## browser_use.llm.openai.like

### ChatOpenAILike

```python
class ChatOpenAILike(ChatOpenAI)
```

Base class for OpenAI-compatible API providers.

A class for interacting with any LLM provider that implements the OpenAI API
schema. This serves as a base class for providers that are compatible with
OpenAI's API format but run on different infrastructure (e.g., Azure OpenAI,
local Ollama servers, or other OpenAI-compatible services).

This class extends ChatOpenAI and inherits all its functionality, allowing
seamless integration with providers that follow the OpenAI API specification.

**Arguments**:

- `model` - The name of the model to use.
- `base_url` - Optional custom API endpoint URL.
- `api_key` - API key for authentication (if required).

**Example**:

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


## browser_use.llm.openai.chat

### ChatOpenAI

```python
class ChatOpenAI(BaseChatModel)
```

OpenAI chat model integration for browser automation.

A wrapper around AsyncOpenAI that implements the BaseLLM protocol.
This class accepts all AsyncOpenAI parameters while adding model
and temperature parameters for the LLM interface (if temperature is not `None`).

Supports all OpenAI chat models including GPT-4, GPT-3.5, and o1 reasoning models.
Handles structured output, vision capabilities, and automatic retries.

Constructor Parameters:
model: Model name ("gpt-4", "gpt-3.5-turbo", "o1-preview", etc.)
api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
temperature: Sampling temperature 0-2 (default: 0.2 for consistency)
frequency_penalty: Reduce repetition -2 to 2 (default: 0.3)
reasoning_effort: For o1 models: "low", "medium", "high" (default: "low")
seed: Deterministic sampling seed
top_p: Nucleus sampling threshold
max_completion_tokens: Max tokens in response (default: 4096)

base_url: Custom API endpoint (for OpenAI-compatible servers)
organization: OpenAI organization ID
project: OpenAI project ID
timeout: Request timeout in seconds
max_retries: Retry attempts on failure (default: 5)
add_schema_to_system_prompt: Include schema in prompt vs response_format

**Example**:

  >>> llm = ChatOpenAI(
  ...     model="gpt-4",
  ...     api_key="sk-...",
  ...     temperature=0.7
  ... )
  >>> agent = Agent(task="Search for docs", llm=llm)

  >>> # Using local OpenAI-compatible server
  >>> llm = ChatOpenAI(
  ...     model="local-model",
  ...     base_url="http://localhost:8000/v1",
  ...     api_key="dummy"
  ... )


## browser_use.llm.exceptions


## browser_use.llm.views

### ChatInvokeUsage

```python
class ChatInvokeUsage(BaseModel)
```

Usage information for a chat model invocation.

A data class holding token usage statistics for a single LLM invocation,
including prompt tokens, completion tokens, cached tokens, and total usage.

### ChatInvokeCompletion

```python
class ChatInvokeCompletion(BaseModel, Generic[T])
```

Response from a chat model invocation.

A data class that holds the completion response from an LLM invocation,
including the actual completion content, usage statistics, and optional
thinking/reasoning information.


## browser_use.llm.schema


## browser_use.llm.ollama.serializer


## browser_use.llm.ollama.chat

### ChatOllama

```python
class ChatOllama(BaseChatModel)
```

Ollama chat model integration for browser automation.

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

**Example**:

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


## browser_use.llm.tests.test_chat_models


## browser_use.llm.tests.test_gemini_image


## browser_use.llm.tests.test_groq_loop


## browser_use.llm.tests.test_single_step


## browser_use.llm.tests.test_anthropic_cache


## browser_use.llm.google.serializer


## browser_use.llm.google


## browser_use.llm.google.chat

### ChatGoogle

```python
class ChatGoogle(BaseChatModel)
```

Google Gemini chat model integration for browser automation.

Provides access to Google's Gemini models including multimodal capabilities,
code execution, and reasoning. Supports Gemini 2.0 Flash, Gemini 2.0 Flash Exp,
and other Gemini family models.

Constructor Parameters:
model: The Gemini model to use
- "gemini-2.0-flash-exp": Latest experimental model (recommended)
- "gemini-2.0-flash": Stable fast model
- "gemini-2.0-flash-lite-preview-02-05": Lightweight preview
- "Gemini-2.0-exp": Experimental features
api_key: Google API key (defaults to GOOGLE_API_KEY env var)
Note: GOOGLE_API_KEY replaces deprecated GEMINI_API_KEY
temperature: Sampling temperature 0-2
top_p: Nucleus sampling threshold
seed: Random seed for deterministic outputs
thinking_budget: Token budget for thinking/reasoning
config: Additional Gemini configuration dict
- tools: Tool definitions
- safety_settings: Content filtering
- system_instruction: System prompt
vertexai: Use Vertex AI instead of Google AI (enterprise)
credentials: Google Cloud credentials object
project: Google Cloud project ID (for Vertex AI)
location: Google Cloud location (for Vertex AI)
http_options: HTTP client configuration

Feature Support:
- Tool Calling: Yes (full function calling)
- Structured Output: Yes (via JSON mode)
- Vision: Yes (excellent multimodal support)
- Context: 1M tokens (Gemini 1.5 Pro)
- Code Execution: Yes (sandboxed Python)
- Reasoning: Yes (with thinking tokens)

API Key Migration:
The environment variable GEMINI_API_KEY has been renamed to GOOGLE_API_KEY.
Update your .env file accordingly:
Old: GEMINI_API_KEY=...
New: GOOGLE_API_KEY=...

**Example**:

  >>> llm = ChatGoogle(
  ...     model="gemini-2.0-flash-exp",
  ...     api_key="...",  # or set GOOGLE_API_KEY env var
  ...     temperature=0.7,
  ... )
  >>> agent = Agent(task="Analyze webpage content", llm=llm)


## browser_use.llm


## browser_use.llm.deepseek.serializer


## browser_use.llm.deepseek.chat

### ChatDeepSeek

```python
class ChatDeepSeek(BaseChatModel)
```

DeepSeek chat model integration for browser automation.

Provides access to DeepSeek's reasoning and coding models with competitive
performance at low cost. Supports DeepSeek-V3, DeepSeek-Chat, and other
DeepSeek models optimized for complex reasoning tasks.

Constructor Parameters:
model: Model name (default: "deepseek-chat")
- "deepseek-chat": General purpose model
- "deepseek-v3": Latest reasoning model
- "deepseek-coder": Code-focused model
api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
temperature: Sampling temperature 0-2
top_p: Nucleus sampling threshold
seed: Random seed for deterministic outputs
max_tokens: Maximum tokens in response
base_url: API endpoint (default: https://api.deepseek.com/v1)
timeout: Request timeout in seconds
client_params: Additional OpenAI client parameters

Feature Support:
- Tool Calling: Yes (full function calling support)
- Structured Output: Yes (via JSON mode)
- Vision: Limited (use_vision=False recommended)
- Context: 64k-128k tokens depending on model
- Strengths: Reasoning, coding, math, cost-effective

Vision Guidance:
DeepSeek models have limited vision capabilities. For tasks requiring
detailed image analysis or screenshots, set use_vision=False or use
ChatOpenAI/ChatAnthropic instead. Basic image understanding may work
but is not optimized for browser automation scenarios.

**Example**:

  >>> llm = ChatDeepSeek(
  ...     model="deepseek-chat",
  ...     api_key="sk-...",
  ...     temperature=0.7,
  ... )
  >>> agent = Agent(
  ...     task="Code analysis and automation",
  ...     llm=llm,
  ...     use_vision=False  # Recommended for DeepSeek
  ... )


## browser_use.llm.groq.parser


## browser_use.llm.groq.serializer


## browser_use.llm.groq.chat

### ChatGroq

```python
class ChatGroq(BaseChatModel)
```

Groq chat model integration for browser automation.

Provides ultra-fast inference with Groq's accelerated LLM infrastructure.
Supports Llama 4, Qwen 3, Kimi K2, and GPT models with high-speed processing
for browser automation tasks.

Constructor Parameters:
model: Model name from Groq's catalog
- "meta-llama/llama-4-maverick-17b-128e-instruct" (recommended)
- "meta-llama/llama-4-scout-17b-16e-instruct"
- "qwen/qwen3-32b"
- "moonshotai/kimi-k2-instruct" (supports tool calling)
- "openai/gpt-oss-20b"
- "openai/gpt-oss-120b"
api_key: Groq API key (defaults to GROQ_API_KEY env var)
temperature: Sampling temperature 0-2 (default: None)
top_p: Nucleus sampling threshold
seed: Random seed for deterministic outputs
service_tier: Performance tier ("auto", "on_demand", "flex")
base_url: Custom API endpoint (for proxies/gateways)
timeout: Request timeout in seconds
max_retries: Retry attempts on failure (default: 10)

Feature Support:
- Structured Output: Llama 4 and GPT models via JSON schema
- Tool Calling: Kimi K2 model only
- Vision: Not supported (use_vision must be False)
- Speed: 10-100x faster than traditional providers
- Context: Varies by model (8k-128k tokens)

Vision Guidance:
Groq models currently do not support vision/image inputs.
For multimodal tasks, use ChatOpenAI, ChatAnthropic, or ChatGoogle instead.

**Example**:

  >>> llm = ChatGroq(
  ...     model="meta-llama/llama-4-maverick-17b-128e-instruct",
  ...     api_key="gsk_...",
  ...     temperature=0.1,
  ... )
  >>> agent = Agent(task="Fast web interactions", llm=llm, use_vision=False)


## browser_use.llm.azure.chat

### ChatAzureOpenAI

```python
class ChatAzureOpenAI(ChatOpenAILike)
```

Azure OpenAI chat model integration for browser automation.

Provides access to OpenAI models through Microsoft Azure's infrastructure
with enhanced security, compliance, and regional availability. Supports
GPT-4, GPT-4 Turbo, and GPT-3.5 models via Azure OpenAI Service.

**Example**:

  >>> llm = ChatAzureOpenAI(
  ...	    model="gpt-4",
  ...	    api_key="...",
  ...	    azure_endpoint="https://your-resource.openai.azure.com",
  ...	    azure_deployment="gpt-4-deployment",
  ... )
  >>> agent = Agent(task="Enterprise web automation", llm=llm)


## browser_use.llm.messages

### ContentPartTextParam

```python
class ContentPartTextParam(BaseModel)
```

Text content part for LLM messages.

A class for handling text parts within a message, used when messages
contain mixed content types (text and images).

### ContentPartRefusalParam

```python
class ContentPartRefusalParam(BaseModel)
```

Refusal content part for LLM messages.

A class for handling refusal messages from the AI assistant when it
cannot or will not complete a requested action.

### ContentPartImageParam

```python
class ContentPartImageParam(BaseModel)
```

Image content part for LLM messages.

A class for handling image parts within messages, enabling vision-based
interactions with LLMs that support image input.

### ToolCall

```python
class ToolCall(BaseModel)
```

Tool call specification with ID and function details.

Represents a tool call generated by the LLM, including the tool ID
and function parameters for execution.

### UserMessage

```python
class UserMessage(_MessageBase)
```

User message with text and optional image content.

Represents a message from the user, used when creating conversation
histories for LLM interactions. Can contain text, images, or both.

### SystemMessage

```python
class SystemMessage(_MessageBase)
```

System message for providing context and instructions.

Represents a system-level instruction message, typically used to set
the behavior, personality, or constraints for the AI assistant.

### AssistantMessage

```python
class AssistantMessage(_MessageBase)
```

Assistant message with text content and optional tool calls.

Represents a message from the AI assistant, including text responses
and any tool calls the assistant wants to execute.


## browser_use.llm.base

### BaseChatModel

```python
class BaseChatModel(Protocol)
```

Protocol defining the interface for all language model implementations.

All LLM implementations (OpenAI, Anthropic, Google, etc.) must conform to
this protocol. This ensures consistent behavior across different providers
and enables type-safe usage throughout the library.

The protocol uses Python's Protocol typing (PEP 544) to define structural
typing - any class that implements these methods and properties is considered
a valid BaseChatModel, without explicit inheritance.

**Attributes**:

- `model` - The model identifier (e.g., 'gpt-4', 'claude-3-opus').
- `provider` - The provider name (e.g., 'openai', 'anthropic').
- `name` - Human-readable name for the model.
- `model_name` - Legacy alias for model attribute.

**Example**:

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

**Notes**:

  This is a Protocol class, not meant to be instantiated directly.
  Use one of the concrete implementations like ChatOpenAI, ChatAnthropic, etc.

**See Also**:

- `ChatOpenAI` - OpenAI implementation
- `ChatAnthropic` - Anthropic/Claude implementation
- `ChatGoogle` - Google/Gemini implementation


## browser_use.llm.openrouter.serializer


## browser_use.llm.openrouter.chat

### ChatOpenRouter

```python
class ChatOpenRouter(BaseChatModel)
```

OpenRouter chat model integration for browser automation.

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

**Example**:

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


## browser_use.llm.anthropic.serializer


## browser_use.llm.anthropic.chat

### ChatAnthropic

```python
class ChatAnthropic(BaseChatModel)
```

Anthropic Claude chat model integration for browser automation.

Provides access to Anthropic's Claude models with advanced reasoning capabilities,
long context understanding, and multimodal processing. Supports Claude 3.5 Sonnet,
Claude 3 Haiku, Claude 3 Opus, and other Claude family models.

Constructor Parameters:
model: Claude model to use
- "claude-3-5-sonnet-latest": Most capable, balanced (recommended)
- "claude-3-5-haiku-latest": Fast and cost-effective
- "claude-3-opus-latest": Highest capability for complex tasks
- "claude-3-sonnet-20240229": Previous Sonnet version
- "claude-3-haiku-20240307": Previous Haiku version

api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)

max_tokens: Maximum tokens in response (default: 8192, max: 8192)
Note: Claude requires explicit max_tokens, unlike OpenAI

temperature: Sampling temperature 0-1 (default: None)
Lower values = more deterministic

top_p: Nucleus sampling threshold (default: None)

seed: Random seed for deterministic outputs (beta feature)

base_url: Custom API endpoint (for proxies/gateways)

timeout: Request timeout in seconds

max_retries: Retry attempts on failure (default: 10)

Feature Support:
- Tool Calling: Yes (native tool use)
- Structured Output: Yes (via tool use or prompting)
- Vision: Yes (all Claude 3+ models)
- Context: 200k tokens (all models)
- Caching: Yes (prompt caching for repeated content)
- Computer Use: Yes (Claude 3.5 Sonnet only, beta)

Prompt Caching:
Claude supports automatic caching of repeated prompt content to reduce
costs and latency. The system automatically marks static content for
caching when the same context is used multiple times.

Vision Capabilities:
All Claude 3+ models support image understanding. The agent automatically
includes screenshots when use_vision=True. Claude excels at:
- UI element detection and interaction
- Text extraction from images
- Visual reasoning and analysis
- Chart and diagram understanding

**Example**:

  >>> # Basic usage
  >>> llm = ChatAnthropic(
  ...     model="claude-3-5-sonnet-latest",
  ...     api_key="sk-ant-...",  # or set ANTHROPIC_API_KEY env var
  ...     max_tokens=8192,
  ... )
  >>> agent = Agent(task="Navigate and fill forms", llm=llm)

  >>> # Fast, cost-effective model
  >>> llm = ChatAnthropic(
  ...     model="claude-3-5-haiku-latest",
  ...     max_tokens=4096,
  ...     temperature=0.3,
  ... )

  >>> # Maximum capability for complex tasks
  >>> llm = ChatAnthropic(
  ...     model="claude-3-opus-latest",
  ...     max_tokens=8192,
  ...     temperature=0.7,
  ... )

  Best Practices:
  - Use Sonnet for balanced performance and capability
  - Use Haiku for high-volume, simpler tasks
  - Use Opus when maximum reasoning is required
  - Always set max_tokens (required parameter)
  - Enable vision for UI automation tasks
  - Leverage prompt caching for repeated contexts


## browser_use.llm.aws.serializer


## browser_use.llm.aws


## browser_use.llm.aws.chat_anthropic

### ChatAnthropicBedrock

```python
class ChatAnthropicBedrock(ChatAWSBedrock)
```

AWS Bedrock Anthropic Claude chat model integration for browser automation.

Provides optimized access to Anthropic Claude models through AWS Bedrock
with Claude-specific defaults and configurations. Supports Claude 3.5 Sonnet,
Claude 3 Opus, Claude 3 Haiku with AWS enterprise features.

**Example**:

  >>> llm = ChatAnthropicBedrock(
  ...	    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
  ...	    aws_region="us-east-1",
  ...	    max_tokens=8192,
  ... )
  >>> agent = Agent(task="Advanced reasoning tasks", llm=llm)


## browser_use.llm.aws.chat_bedrock

### ChatAWSBedrock

```python
class ChatAWSBedrock(BaseChatModel)
```

AWS Bedrock chat model integration for browser automation.

Provides access to multiple foundation models through AWS Bedrock including
Anthropic Claude, Meta Llama, Cohere Command, and Amazon Titan models.
Offers enterprise-grade security, compliance, and serverless scaling.

Constructor Parameters:
model: Bedrock model ID (default: "anthropic.claude-3-5-sonnet-20240620-v1:0")
- Anthropic: "anthropic.claude-3-5-sonnet-20240620-v1:0"
- Meta: "meta.llama3-2-90b-instruct-v1:0"
- Cohere: "cohere.command-r-plus-v1:0"
- Amazon: "amazon.titan-text-premier-v1:0"
max_tokens: Maximum tokens in response (default: 4096)
temperature: Sampling temperature 0-1
top_p: Nucleus sampling threshold
seed: Random seed for deterministic outputs
stop_sequences: Custom stop sequences

AWS Authentication (in order of preference):
1. session: Pre-configured boto3 Session object
2. aws_sso_auth: Use AWS SSO (set to True)
3. IAM Role: Automatic when running on EC2/Lambda/ECS
4. Environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION or AWS_DEFAULT_REGION
5. Constructor parameters:
- aws_access_key_id
- aws_secret_access_key
- aws_region

request_params: Additional Bedrock request parameters

IAM Authentication Best Practice:
When running on AWS infrastructure (EC2, Lambda, ECS), prefer IAM roles
over explicit credentials. The SDK automatically uses the instance role:
>>> llm = ChatAWSBedrock(
...     model="anthropic.claude-3-5-sonnet-20240620-v1:0",
...     aws_region="us-east-1"  # Only region needed with IAM role
... )

Regions:
Bedrock is available in: us-east-1, us-west-2, eu-west-1, eu-central-1,
ap-northeast-1, ap-southeast-2, and others. Check AWS docs for model availability.

**Example**:

  >>> # Using environment variables
  >>> llm = ChatAWSBedrock(
  ...     model="anthropic.claude-3-5-sonnet-20240620-v1:0",
  ...     aws_region="us-east-1",
  ...     max_tokens=4096,
  ... )
  >>> agent = Agent(task="Enterprise web automation", llm=llm)


## browser_use.dom.enhanced_snapshot


## browser_use.dom.utils


## browser_use.dom.views

### CurrentPageTargets

```python
class CurrentPageTargets
```

Container for current page and iframe target information.

This class holds information about the current page and all iframes across
all pages in the browser session. Used internally for managing CDP sessions
and cross-origin iframe communication.

**Attributes**:

- `page_session` - TargetInfo for the main page
- `iframe_sessions` - List of TargetInfo for all iframes (not just current page)

**Notes**:

  Iframe sessions are ALL the iframe sessions of all the pages, not just
  the current page. This is important for proper session management.

**Example**:

  >>> targets = await browser_session._get_current_page_targets()
  >>> print(f"Main page: {targets.page_session.url}")
  >>> print(f"Iframes: {len(targets.iframe_sessions)}")

### DOMRect

```python
class DOMRect
```

Represents a rectangular area in the DOM with position and dimensions.

Provides position and dimension information for DOM elements.
Used in EnhancedDOMTreeNode.absolute_position to describe element bounds.

**Attributes**:

- `x` - The x-coordinate of the rectangle's left edge in pixels
- `y` - The y-coordinate of the rectangle's top edge in pixels
- `width` - The width of the rectangle in pixels
- `height` - The height of the rectangle in pixels

**Example**:

  >>> element = await browser_session.get_element_by_index(5)
  >>> if element.absolute_position:
  ...     pos = element.absolute_position
  ...     print(f"Element at ({pos.x}, {pos.y})")
  ...     print(f"Size: {pos.width}x{pos.height}")

### EnhancedDOMTreeNode

```python
class EnhancedDOMTreeNode
```

Enhanced DOM tree node that contains information from AX, DOM, and Snapshot trees.

This is the primary data structure for representing DOM elements in browser-use 0.7.
It combines data from Chrome DevTools Protocol's DOM, Accessibility, and Snapshot
trees into a single unified structure. This node type is returned from selector_map
in BrowserStateSummary and replaces Playwright's ElementHandle.

Key Attributes:
element_index: Unique index for this element in the current DOM state
node_id: CDP DOM node ID for this element
backend_node_id: CDP backend node ID (persistent across navigations)
node_type: Type of DOM node (ELEMENT_NODE, TEXT_NODE, etc.)
node_name: Tag name for element nodes (e.g., 'DIV', 'INPUT')
node_value: Text content for text nodes
attributes: Dict of HTML attributes and their values
is_visible: Whether the element is currently visible on the page
is_scrollable: Whether the element can be scrolled
absolute_position: DOMRect with absolute position on the page
target_id: CDP target ID for the frame containing this element
frame_id: Frame ID if this element is in an iframe
session_id: CDP session ID for cross-origin iframes
tag_name: Convenience property for node_name in uppercase

Navigation Properties:
parent_node: Reference to parent node in the tree
children_nodes: List of child nodes
content_document: For iframe elements, the document node inside
shadow_roots: List of shadow root nodes if present

Accessibility Properties:
ax_node: Enhanced accessibility node with ARIA properties

Snapshot Properties:
snapshot_node: Enhanced snapshot data with layout information

Useful Methods:
get_all_children_text(): Get all text content from element and children
is_actually_scrollable: Enhanced scroll detection property

Learn more about the underlying CDP fields:
- (DOM node) https://chromedevtools.github.io/devtools-protocol/tot/DOM/`type`-BackendNode
- (AX node) https://chromedevtools.github.io/devtools-protocol/tot/Accessibility/`type`-AXNode
- (Snapshot node) https://chromedevtools.github.io/devtools-protocol/tot/DOMSnapshot/`type`-DOMNode

**Example**:

  >>> # Get element from browser state (replaces wait_for_selector)
  >>> state = await browser_session.get_browser_state_summary()
  >>> for index, element in state.dom_state.selector_map.items():
  ...     if element.tag_name == 'BUTTON' and 'Submit' in element.get_all_children_text():
  ...         # Found the element, now interact with it
  ...         node = await browser_session.get_element_by_index(index)
  ...         await browser_session.event_bus.dispatch(ClickElementEvent(node=node))
  ...         break
  >>>
  >>> # Check element properties
  >>> node = await browser_session.get_element_by_index(5)
  >>> print(f"Tag: {node.tag_name}")
  >>> print(f"Visible: {node.is_visible}")
  >>> print(f"Text: {node.get_all_children_text()}")
  >>> print(f"Attributes: {node.attributes}")

#### EnhancedDOMTreeNode.parent

```python
@property
def parent(self) -> 'EnhancedDOMTreeNode | None'
```

Get the parent node in the DOM tree.

Returns the parent node of this element in the DOM hierarchy.
Returns None if this is a root node or has no parent.

**Returns**:

  The parent EnhancedDOMTreeNode or None.

**Example**:

  >>> element = await browser_session.get_element_by_index(10)
  >>> parent = element.parent
  >>> if parent:
  ...     print(f"Parent tag: {parent.tag_name}")

#### EnhancedDOMTreeNode.children

```python
@property
def children(self) -> list['EnhancedDOMTreeNode']
```

Get all direct child nodes.

Returns a list of all direct child nodes of this element.
Does not include shadow roots (use children_and_shadow_roots for that).

**Returns**:

  List of child EnhancedDOMTreeNode objects.

**Example**:

  >>> element = await browser_session.get_element_by_index(5)
  >>> for child in element.children:
  ...     if child.tag_name == 'button':
  ...         print(f"Found button child: {child.element_index}")

#### EnhancedDOMTreeNode.tag_name

```python
@property
def tag_name(self) -> str
```

Get the lowercase tag name of this element.

Returns the HTML tag name in lowercase (e.g., 'div', 'input', 'button').
This is a convenience property for node_name.lower().

**Returns**:

  Lowercase tag name string.

**Example**:

  >>> element = await browser_session.get_element_by_index(7)
  >>> if element.tag_name == 'input':
  ...     print("Found an input field")

#### EnhancedDOMTreeNode.xpath

```python
@property
def xpath(self) -> str
```

Generate XPath for this DOM node, stopping at shadow boundaries or iframes.

Constructs an XPath selector that uniquely identifies this element
within its document context. The path stops at shadow DOM boundaries
or iframe boundaries since these create separate document contexts.

**Returns**:

  XPath string for this element.

**Example**:

  >>> element = await browser_session.get_element_by_index(12)
  >>> print(f"Element XPath: {element.xpath}")
  >>> # Output: /html/body/div[2]/form/input[1]

#### EnhancedDOMTreeNode.get_all_children_text

```python
def get_all_children_text(self, max_depth: int = -1) -> str
```

Recursively get all text content from this node and its children.

Extracts all text content from this element and its descendants.
Useful for getting the complete text within a container element.

**Arguments**:

- `max_depth` - Maximum depth to traverse (-1 for unlimited).

**Returns**:

  Combined text content from all descendant text nodes.

**Example**:

  >>> element = await browser_session.get_element_by_index(10)
  >>> text = element.get_all_children_text()
  >>> print(f"Element text: {text}")

#### EnhancedDOMTreeNode.is_actually_scrollable

```python
@property
def is_actually_scrollable(self) -> bool
```

Enhanced scroll detection that combines CDP detection with CSS analysis.

This detects scrollable elements that Chrome's CDP might miss, which is common
in iframes and dynamically sized containers. It performs more thorough
checks than the basic is_scrollable property.

**Returns**:

  True if the element can be scrolled, False otherwise.

**Example**:

  >>> element = await browser_session.get_element_by_index(15)
  >>> if element.is_actually_scrollable:
  ...     # Element can be scrolled
  ...     tools = Tools()
  ...     await tools.act(
  ...         action=ActionModel(scroll=ScrollAction(
  ...             down=True,
  ...             num_pages=1.0,
  ...             frame_element_index=element.element_index
  ...         )),
  ...         browser_session=browser_session
  ...     )

#### EnhancedDOMTreeNode.scroll_info

```python
@property
def scroll_info(self) -> dict[str, Any] | None
```

Calculate scroll information for this element if it's scrollable.

Provides detailed scroll metrics including position, pages above/below,
and scroll percentages. Only available for scrollable elements.

**Returns**:

  Dictionary with scroll metrics or None if not scrollable.
  Keys include:
  - pages_above: Number of viewport pages scrolled above
  - pages_below: Number of viewport pages available below
  - vertical_scroll_percentage: Current vertical scroll position (0-100)
  - horizontal_scroll_percentage: Current horizontal scroll position (0-100)
  - can_scroll_down: Whether more content is available below
  - can_scroll_up: Whether content is available above
  - can_scroll_left: Whether content is available to the left
  - can_scroll_right: Whether content is available to the right

**Example**:

  >>> element = await browser_session.get_element_by_index(20)
  >>> if element.scroll_info:
  ...     info = element.scroll_info
  ...     print(f"Pages below: {info['pages_below']:.1f}")
  ...     print(f"Scroll position: {info['vertical_scroll_percentage']:.0f}%")
  ...     if info['can_scroll_down']:
  ...         print("Can scroll down for more content")

### SerializedDOMState

```python
class SerializedDOMState
```

Serialized DOM state for LLM consumption.

Contains DOM tree representation and selector map for element interaction.
This is returned as part of BrowserStateSummary from get_browser_state_summary()
and is the primary data structure for accessing DOM elements in browser-use 0.7.

**Attributes**:

- `selector_map` - Dictionary mapping element indices to EnhancedDOMTreeNode objects.
  This is the main way to access interactive elements by their indices.
  Keys are integer indices, values are EnhancedDOMTreeNode instances.
- `_root` - Internal simplified node tree (not for direct use)

**Methods**:

- `llm_representation(include_attributes)` - Get string representation of DOM tree
  for LLM processing. This returns the serialized DOM tree as text.

**Example**:

  >>> # Get browser state and access interactive elements
  >>> state = await browser_session.get_browser_state_summary()
  >>>
  >>> # Iterate through all interactive elements
  >>> for index, element in state.dom_state.selector_map.items():
  ...     print(f"[{index}] {element.tag_name}: {element.get_all_children_text()}")
  >>>
  >>> # Find specific elements for interaction (replaces wait_for_selector)
  >>> button_element = None
  >>> for index, element in state.dom_state.selector_map.items():
  ...     if element.tag_name == 'BUTTON' and 'Submit' in element.get_all_children_text():
  ...         button_element = await browser_session.get_element_by_index(index)
  ...         break
  >>>
  >>> # Get DOM representation for LLM
  >>> dom_text = state.dom_state.llm_representation()
  >>> print(f"DOM tree: {dom_text[:500]}...")  # First 500 chars

### DOMInteractedElement

```python
class DOMInteractedElement
```

Represents a DOM element that has been interacted with by the agent.

This class captures information about DOM elements that the agent has
interacted with (clicked, typed into, etc.) during a browser session.
It stores a snapshot of the element's state at the time of interaction
for history tracking and debugging purposes.

**Attributes**:

- `node_id` - CDP DOM node ID of the interacted element
- `backend_node_id` - CDP backend node ID (persistent across navigations)
- `frame_id` - Frame ID if element is in an iframe, None otherwise
- `node_type` - Type of DOM node (ELEMENT_NODE, TEXT_NODE, etc.)
- `node_value` - Text content for text nodes
- `node_name` - Tag name for element nodes (e.g., 'DIV', 'INPUT')
- `attributes` - Dictionary of HTML attributes at interaction time
- `bounds` - DOMRect with element position and dimensions
- `x_path` - XPath selector to the element
- `element_hash` - Hash value for element identification

**Example**:

  >>> # Element is captured when interaction occurs
  >>> state = await browser_session.get_browser_state_summary()
  >>> for interaction in state.interacted_element:
  ...     if interaction:
  ...         print(f"Interacted with: {interaction.node_name}")
  ...         print(f"XPath: {interaction.x_path}")


## browser_use.dom.playground.tree


## browser_use.dom.playground.multi_act


## browser_use.dom.playground.extraction


## browser_use.dom.service

### DomService

```python
class DomService
```

Service for getting the DOM tree and other DOM-related information.

The DomService is responsible for DOM serialization, element querying, and
maintaining interactive element indices. It provides the core functionality
for converting web page content into a format the agent can understand and
interact with.

Key Responsibilities:
- DOM tree serialization with element indexing
- Clickable element detection and hashing
- Cross-origin iframe handling
- Element querying and selection
- Accessibility tree integration

Key Methods:
get_dom_state(): Serialize entire DOM with interactive elements
click_element(): Click element by index
input_text(): Type text into element by index
get_element_by_index(): Retrieve element by its index
select_option(): Select dropdown option by index

Performance Considerations:
- DOM serialization is cached per page state
- Element indices are deterministic but change on DOM mutations
- Cross-origin iframes require additional CDP connections

**Arguments**:

- `browser_session` - The browser session to operate on
- `logger` - Optional logger instance
- `cross_origin_iframes` - Whether to traverse cross-origin iframes (slower)

**Notes**:

  Currently creates a new websocket connection per step. Future optimization
  will maintain persistent connections for better performance.


## browser_use.dom.serializer.serializer


## browser_use.dom.serializer.clickable_elements


## browser_use.dom.debug.highlights


## browser_use.integrations.gmail.actions

### register_gmail_actions

```python
def register_gmail_actions(tools: Tools, gmail_service: GmailService | None = None, access_token: str | None = None) -> Tools
```

Register Gmail actions with the provided tools.

Registers Gmail-related actions (like get_recent_emails) with the browser-use
Tools registry, enabling agents to interact with Gmail during automation tasks.

**Arguments**:

- `tools` - The browser-use tools to register actions with
- `gmail_service` - Optional pre-configured Gmail service instance
- `access_token` - Optional direct access token (alternative to file-based auth)

**Returns**:

- `Tools` - The tools instance with Gmail actions registered.

**Example**:

  >>> from browser_use import Tools
  >>> from browser_use.integrations.gmail import register_gmail_actions
  >>> tools = Tools()
  >>> tools = register_gmail_actions(tools)
  >>> # Now the agent can use get_recent_emails action


## browser_use.integrations.gmail


## browser_use.integrations.gmail.service

### GmailService

```python
class GmailService
```

Gmail API service for email reading.

Provides functionality to:
- Authenticate with Gmail API using OAuth2
- Read recent emails with filtering
- Return full email content for agent analysis

#### GmailService.is_authenticated

```python
def is_authenticated(self) -> bool
```

Check if Gmail service is authenticated.

**Returns**:

- `bool` - True if the Gmail service is authenticated and ready to use.

**Example**:

  >>> gmail = GmailService()
  >>> if not gmail.is_authenticated():
  ...     await gmail.authenticate()

#### GmailService.authenticate

```python
async def authenticate(self) -> bool
```

Handle OAuth authentication and token management.

Initiates the OAuth authentication flow for Gmail API access. Will use
existing tokens if available, refresh expired tokens, or start a new
OAuth flow if needed.

**Returns**:

- `bool` - True if authentication successful, False otherwise.

**Example**:

  >>> gmail = GmailService()
  >>> success = await gmail.authenticate()
  >>> if success:
  ...     emails = await gmail.get_recent_emails()


## browser_use.filesystem


## browser_use.filesystem.file_system

### FileSystem

```python
class FileSystem
```

Enhanced file system with in-memory storage and multiple file type support.

FileSystem provides a sandboxed file management interface for agents to read,
write, and manipulate files during browser automation tasks. It maintains an
in-memory cache synchronized with disk storage.

Key Features:
- Sandboxed file operations in a dedicated directory
- Support for multiple file types (txt, md, json, csv, pdf)
- In-memory caching with disk synchronization
- Automatic PDF conversion from markdown
- Safe filename validation and parsing

Methods Exposed to Tools:
read_text(filename): Read file content as string
write_text(filename, content): Write content to file
append_text(filename, content): Append content to existing file
replace_in_file(filename, old_str, new_str): Replace text in file
list_files(): Get list of available files
available_file_paths: Property returning list of file paths

File Availability:
Files are available to tools via the available_file_paths property,
which is automatically injected into tool functions. Only files in
the sandboxed directory are accessible.

Supported Extensions:
.txt: Plain text files
.md: Markdown files
.json: JSON data files
.csv: CSV spreadsheet files
.pdf: PDF documents (write markdown, auto-converts to PDF)

Security:
- All operations are restricted to the sandboxed directory
- Filenames must be alphanumeric with supported extensions
- Path traversal attempts are blocked
- External file access requires explicit permission

**Example**:

  >>> fs = FileSystem("./agent_workspace")
  >>> await fs.write_text("notes.md", "# Meeting Notes")
  >>> content = await fs.read_text("notes.md")
  >>> files = fs.list_files()  # Returns ["todo.md", "notes.md"]


## browser_use.logging_config

### setup_logging

```python
def setup_logging(stream=None, log_level=None, force_setup=False, debug_log_file=None, info_log_file=None)
```

Setup logging configuration for browser-use.

Configures the application's logging system with colored output, custom log levels,
and optional file handlers for debugging.

**Arguments**:

- `stream` - Output stream for logs (default: sys.stdout). Can be sys.stderr for MCP mode.
- `log_level` - Override log level (default: uses CONFIG.BROWSER_USE_LOGGING_LEVEL)
- `force_setup` - Force reconfiguration even if handlers already exist
- `debug_log_file` - Path to log file for debug level logs only
- `info_log_file` - Path to log file for info level logs only

**Returns**:

- `logging.Logger` - The configured browser_use logger.

**Example**:

  >>> from browser_use.logging_config import setup_logging
  >>> logger = setup_logging(log_level='debug')
  >>> logger.info('Browser-use initialized')


## browser_use.telemetry.views


## browser_use.telemetry


## browser_use.telemetry.service

### ProductTelemetry

```python
class ProductTelemetry
```

Service for capturing anonymized telemetry data.

Collects anonymous usage statistics to help improve browser-use. No personal
data, credentials, or page content is ever collected - only feature usage
patterns and performance metrics.

Data Collected:
- Feature usage (which tools/actions are used)
- Performance metrics (execution times, step counts)
- Error types (not content)
- System info (OS, Python version)
- Anonymous device ID (random UUID)

NOT Collected:
- URLs visited
- Page content or screenshots
- Credentials or sensitive data
- Task descriptions
- LLM prompts or responses

Storage:
Device ID stored in: ~/.browser-use/device_id
Data sent to: PostHog (EU servers)
Retention: 90 days

Disable Telemetry:
Set environment variable: ANONYMIZED_TELEMETRY=false
Or in code: os.environ['ANONYMIZED_TELEMETRY'] = 'false'

**Notes**:

  Telemetry helps us understand usage patterns and improve the library.
  It's completely anonymous and can be disabled at any time.


## browser_use.tools.registry.views

### ActionModel

```python
class ActionModel(BaseModel)
```

Base model for dynamically created action models.

This is the container model that holds any single action to be executed.
It's dynamically generated to include all registered actions, where each
field represents a different action type. Only one action field should be
set (non-None) at a time.

The ActionModel is created at runtime based on registered actions and
provides a uniform interface for the agent to specify what action to perform.

**Example**:

  >>> # Click on element with index 5
  >>> action = ActionModel(click_element_by_index=ClickElementAction(index=5))
  >>>
  >>> # Navigate to a URL
  >>> action = ActionModel(go_to_url=GoToUrlAction(url="https://example.com"))
  >>>
  >>> # Input text into element 3
  >>> action = ActionModel(input_text=InputTextAction(index=3, text="Hello"))

**Notes**:

  The exact fields available depend on which actions are registered
  with the Tools registry. Use Tools.get_action_model() to get the
  current ActionModel class with all available actions.


## browser_use.tools.registry.service

### Registry

```python
class Registry(Generic[Context])
```

Service for registering and managing actions.

The Registry manages all available browser actions that can be executed
by the agent. It provides registration, discovery, and execution of actions,
handling parameter injection and validation automatically.

This class is typically accessed through Tools.registry rather than directly.

Key Responsibilities:
- Action registration via decorator pattern
- Parameter validation and injection
- Dynamic ActionModel generation
- Domain-based action filtering
- Telemetry tracking for executed actions

Type Parameters:
Context: Type of custom context data passed to actions

**Example**:

  >>> from browser_use import Tools
  >>> tools = Tools()
  >>>
  >>> # Register custom action via tools.registry
  >>> @tools.registry.action("Extract product price")
  >>> async def extract_price(browser_session) -> ActionResult:
  ...     # Implementation
  ...     return ActionResult(extracted_content="$99.99")

#### Registry.action

```python
def action(self, description: str, param_model: type[BaseModel] | None = None, domains: list[str] | None = None, allowed_domains: list[str] | None = None)
```

Decorator for registering actions.

An alternative way to access the action decorator through the registry.
Used to register custom actions directly with the registry object.

**Arguments**:

- `description` - Description of what the action does
- `param_model` - Optional Pydantic model for action parameters
- `domains` - List of allowed domains for this action
- `allowed_domains` - Alias for domains parameter

**Returns**:

  Decorator function that registers the action

**Example**:

  >>> from browser_use import Tools
  >>> tools = Tools()
  >>> @tools.registry.action("Search for text on page")
  ... async def search_text(text: str) -> ActionResult:
  ...     return ActionResult(...)


## browser_use.tools.views

### SearchGoogleAction

```python
class SearchGoogleAction(BaseModel)
```

Action model for performing a Google search.

Performs a Google search with the specified query. The browser will
navigate to Google and execute the search.

**Arguments**:

- `query` - The search query string to execute on Google.

**Example**:

  >>> action = SearchGoogleAction(query="browser automation python")

### GoToUrlAction

```python
class GoToUrlAction(BaseModel)
```

Action model for navigating to a URL.

Navigates the browser to the specified URL, either in the current tab
or a new tab.

**Arguments**:

- `url` - The URL to navigate to.
- `new_tab` - Whether to open the URL in a new tab (True) or navigate in current tab (False).

**Example**:

  >>> # Navigate in current tab
  >>> action = GoToUrlAction(url="https://example.com")
  >>>
  >>> # Open in new tab
  >>> action = GoToUrlAction(url="https://example.com", new_tab=True)

### ClickElementAction

```python
class ClickElementAction(BaseModel)
```

Action model for clicking on a page element.

Clicks on an element identified by its index in the DOM. Elements are
indexed starting from 1 based on their appearance in the interactive
elements list.

**Arguments**:

- `index` - The index of the element to click (1-based indexing).
- `while_holding_ctrl` - Whether to hold Ctrl while clicking to open links in new background tabs.

**Example**:

  >>> # Click the 5th interactive element
  >>> action = ClickElementAction(index=5)
  >>>
  >>> # Click with Ctrl held to open in new tab
  >>> action = ClickElementAction(index=3, while_holding_ctrl=True)

### InputTextAction

```python
class InputTextAction(BaseModel)
```

Action model for inputting text into a page element.

Inputs text into a form field or text area. Can either replace existing
text or append to it.

**Arguments**:

- `index` - The index of the element to input text into (0 for page, 1+ for specific elements).
- `text` - The text content to input.
- `clear_existing` - Whether to clear existing text (True) or append to it (False).

**Example**:

  >>> # Type into the 3rd input field, replacing existing text
  >>> action = InputTextAction(index=3, text="Hello World")
  >>>
  >>> # Append to existing text in field 2
  >>> action = InputTextAction(index=2, text=" additional text", clear_existing=False)

### DoneAction

```python
class DoneAction(BaseModel)
```

Action model for marking a task as complete.

Signals that a task or automation sequence has been completed.
Used to indicate success/failure and provide a summary of results.

**Arguments**:

- `text` - Description or summary of what was accomplished.
- `success` - Whether the task completed successfully.
- `files_to_display` - Optional list of file paths to display or reference.

**Example**:

  >>> # Task completed successfully
  >>> action = DoneAction(
  ...     text="Successfully submitted the form",
  ...     success=True
  ... )
  >>>
  >>> # Task failed with explanation
  >>> action = DoneAction(
  ...     text="Could not find the submit button",
  ...     success=False
  ... )

### StructuredOutputAction

```python
class StructuredOutputAction(BaseModel, Generic[T])
```

Generic action model for returning structured data.

Used when an agent is configured with a structured output model. This action
replaces the standard 'done' action to return typed, validated data according
to a Pydantic model schema.

Type Parameters:
T: The Pydantic model type for the structured output

**Arguments**:

- `success` - Whether the operation was successful.
- `data` - The structured data of type T to return.

**Example**:

  >>> from pydantic import BaseModel
  >>>
  >>> class ProductInfo(BaseModel):
  ...     name: str
  ...     price: float
  ...     in_stock: bool
  >>>
  >>> # Tools automatically uses StructuredOutputAction when output_model is set
  >>> tools = Tools(output_model=ProductInfo)
  >>>
  >>> # Agent will use: StructuredOutputAction[ProductInfo]
  >>> # with data matching the ProductInfo schema

### SwitchTabAction

```python
class SwitchTabAction(BaseModel)
```

Action model for switching to a different browser tab.

Switches browser focus to a different tab, identified either by URL
pattern or by its Tab ID (last 4 characters of the target ID).

**Arguments**:

- `url` - URL or URL substring of the tab to switch to.
- `tab_id` - Exact 4-character Tab ID to match instead of URL.

**Example**:

  >>> # Switch by URL pattern
  >>> action = SwitchTabAction(url="example.com")
  >>>
  >>> # Switch by exact Tab ID
  >>> action = SwitchTabAction(tab_id="A1B2")

### CloseTabAction

```python
class CloseTabAction(BaseModel)
```

Action model for closing a browser tab.

Closes a specific browser tab identified by its Tab ID.

**Arguments**:

- `tab_id` - The 4-character Tab ID of the tab to close (last 4 chars of TargetID).

**Example**:

  >>> # Close tab with ID "A1B2"
  >>> action = CloseTabAction(tab_id="A1B2")

### ScrollAction

```python
class ScrollAction(BaseModel)
```

Action model for scrolling the page or a specific element.

Scrolls the page or a specific scrollable element by the specified amount.

**Arguments**:

- `down` - Direction to scroll - True for down, False for up.
- `num_pages` - Number of pages to scroll (e.g., 0.5 = half page, 1.0 = one page).
- `frame_element_index` - Optional element index to find scroll container for.

**Example**:

  >>> # Scroll down one page
  >>> action = ScrollAction(down=True, num_pages=1.0)
  >>>
  >>> # Scroll up half a page
  >>> action = ScrollAction(down=False, num_pages=0.5)
  >>>
  >>> # Scroll a specific element
  >>> action = ScrollAction(down=True, num_pages=2.0, frame_element_index=10)

### SendKeysAction

```python
class SendKeysAction(BaseModel)
```

Action model for sending keyboard input.

Sends keyboard keys or key combinations to the browser. Useful for
shortcuts, special keys, or text input without a specific target element.

**Arguments**:

- `keys` - The keyboard keys/key combinations to send.

**Example**:

  >>> # Send Enter key
  >>> action = SendKeysAction(keys="Enter")
  >>>
  >>> # Send keyboard shortcut
  >>> action = SendKeysAction(keys="cmd+a")
  >>>
  >>> # Type text directly
  >>> action = SendKeysAction(keys="Hello World")

### UploadFileAction

```python
class UploadFileAction(BaseModel)
```

Action model for uploading a file to a form element.

Uploads a file to a file input element on the page.

**Arguments**:

- `index` - The index of the file input element to upload to.
- `path` - The file system path to the file to upload.

**Example**:

  >>> # Upload a file to the 2nd file input
  >>> action = UploadFileAction(
  ...     index=2,
  ...     path="/path/to/document.pdf"
  ... )

### ExtractPageContentAction

```python
class ExtractPageContentAction(BaseModel)
```

Action model for extracting content from the current page.

Extracts specific content from the current page based on a selector
or extraction specification.

**Arguments**:

- `value` - The content extraction specification or selector.

**Example**:

  >>> # Extract specific content
  >>> action = ExtractPageContentAction(value=".article-content")

### NoParamsAction

```python
class NoParamsAction(BaseModel)
```

Action model for actions that don't require parameters.

Accepts absolutely anything in the incoming data and discards it,
so the final parsed model is empty. Used for actions like go_back()
that don't need any input parameters.

This model is used internally by the framework for parameter-less actions.

**Example**:

  >>> # Used internally by actions like:
  >>> action = ActionModel(go_back=NoParamsAction())

### GetDropdownOptionsAction

```python
class GetDropdownOptionsAction(BaseModel)
```

Action model for retrieving available options from a dropdown element.

Retrieves all available options from a dropdown/select element.

**Arguments**:

- `index` - The index of the dropdown element to get options for (1-based indexing).

**Example**:

  >>> # Get options from the 3rd dropdown
  >>> action = GetDropdownOptionsAction(index=3)

### SelectDropdownOptionAction

```python
class SelectDropdownOptionAction(BaseModel)
```

Action model for selecting an option from a dropdown element.

Selects a specific option from a dropdown/select element.

**Arguments**:

- `index` - The index of the dropdown element (1-based indexing).
- `text` - The text or exact value of the option to select.

**Example**:

  >>> # Select an option from the 2nd dropdown
  >>> action = SelectDropdownOptionAction(
  ...     index=2,
  ...     text="United States"
  ... )


## browser_use.tools.service

### Tools

```python
class Tools(Generic[Context])
```

Tool management service for handling browser actions and structured output.

The Tools class manages the registry of available actions that agents can perform.
It provides default browser actions (navigation, clicking, typing, etc.) and
allows registration of custom actions.

Default Built-in Tools:
Navigation:
search_google(query: str): Search query in Google
- Opens in existing Google tab or agent's about:blank tab if available
- Otherwise opens in new tab
go_to_url(url: str, new_tab: bool=False): Navigate to URL
- new_tab=True: Opens in new tab
- new_tab=False: Navigates current tab
- Handles network errors gracefully
go_back(): Navigate back in browser history
wait(seconds: float=3): Wait for page load
- Default: 3 seconds, Max: 10 seconds
- Use for dynamic content loading

Interaction:
click_element_by_index(index: int, while_holding_ctrl: bool=False): Click element by index
- while_holding_ctrl=True: Opens link in new tab (Ctrl+Click)
- Validates index exists in current DOM
- Scrolls element into view before clicking
input_text(index: int, text: str): Type text into input field
- Clears existing content first
- Only works on input/textarea elements
upload_file_to_element(index: int, filename: str): Upload file to file input
- File must exist in available_file_paths
- Validates element is file input type
select_dropdown_option(index: int, option_text: str): Select dropdown option
- Works with <select>, ARIA menus, custom dropdowns
- Matches by exact text
get_dropdown_options(index: int): Get list of dropdown options
- Returns all available option values
switch_tab(tab_id: str): Switch to specific tab
- Use tab IDs from browser_state.tabs
close_tab(tab_id: str): Close specific tab
send_keys(keys: str): Send special keys or shortcuts
- Examples: "Escape", "Enter", "Control+o", "Control+Shift+T"

DOM Index Note:
Element indices are generated from the current DOM state and change
when the page updates. Indices are deterministic for a given DOM
state but become invalid after navigation or DOM mutations. Always
use fresh browser_state to get current indices.

Index-Hash Behavior:
The system uses content hashing to maintain index stability when
possible. If an element's content hash matches a previously seen
element, it retains the same index even after DOM updates.

Scrolling:
scroll(down: bool=True, num_pages: float=1.0, index: int=None): Scroll page
- down=True: Scroll down, down=False: Scroll up
- num_pages: How many pages (0.5 for half, 1.0 for full)
- index: Optional element index to scroll within
scroll_to_text(text: str): Scroll to specific text on page
- Finds and scrolls to first occurrence

Content:
extract_content(query: str): Extract specific information from page
- Uses LLM to extract structured data based on query
- Returns semantic information from page content

File System:
write_file(file_name: str, content: str): Write content to file
- Supports: .md, .txt, .json, .csv, .pdf
- PDF files: Write markdown, auto-converts to PDF
read_file(file_name: str): Read file contents
replace_in_file(file_name: str, old_str: str, new_str: str): Replace text
- old_str must match exactly
- Good for updating todo items

Control:
done(text: str, success: bool=True, files_to_display: list=[]): Complete task
- text: Summary of results for user
- success: Whether task completed successfully
- files_to_display: Files to show user

Type Parameters:
Context: Type of context data passed to tool actions.

Special Injected Parameters:
Tool functions can use these special parameter names for automatic injection:
- browser_session: BrowserSession - Browser control interface
- context: Any - Custom context from Agent(context=...)
- page_url: str - Current page URL
- cdp_client: CDPClient - Direct CDP access
- page_extraction_llm: BaseChatModel - LLM for content extraction
- file_system: FileSystem - File operations interface
- available_file_paths: list[str] - Files available for upload
- has_sensitive_data: bool - Whether sensitive data is present

When to Create a Custom Tool:
Create a custom tool when you need to:
- Integrate with external APIs or services
- Implement complex multi-step workflows as atomic operations
- Add domain-specific actions not covered by defaults
- Enforce business logic or validation rules
- Handle special authentication or security requirements

Use prompt engineering instead when:
- The task can be done with existing tools
- You just need different behavior occasionally
- The logic is simple and context-dependent

**Example**:

  >>> tools = Tools()
  >>> # Register action with injected parameters
  >>> @tools.registry.action("Extract with AI")
  ... async def extract(
  ...     query: str,  # User parameter
  ...     browser_session: BrowserSession,  # Injected
  ...     page_extraction_llm: BaseChatModel  # Injected
  ... ):
  ...     state = await browser_session.get_browser_state_summary()
  ...     result = await page_extraction_llm.generate(f"Extract {query} from {state.dom_state}")
  ...     return ActionResult(extracted_content=result)

#### Tools.use_structured_output_action

```python
def use_structured_output_action(self, output_model: type[T])
```

Register a structured output action with the given model type.

Replaces the default 'done' action with a structured output action that
validates and returns data according to the provided Pydantic model.

This method internally calls _register_done_action() to replace the default
done action with one that expects structured output matching the model schema.

**Arguments**:

- `output_model` - A Pydantic model class defining the expected output structure.
  The agent will return an instance of this model when completing tasks.

**Example**:

  >>> from pydantic import BaseModel
  >>> class SearchResult(BaseModel):
  ...     title: str
  ...     url: str
  ...     summary: str
  >>>
  >>> tools = Tools()
  >>> tools.use_structured_output_action(SearchResult)
  >>> # Now the agent will return SearchResult instances

#### Tools.action

```python
def action(self, description: str, **kwargs)
```

Decorator for registering custom actions.

The primary decorator for defining a new custom action that can be used by
the agent. Actions are functions that the agent can call to interact with
the browser or perform other tasks.

**Arguments**:

- `description` - Describe the LLM what the function does (better description == better function calling)
- `**kwargs` - Additional parameters for action configuration

**Returns**:

  Decorator function that registers the action

**Example**:

  >>> tools = Tools()
  >>> @tools.action("Click the submit button")
  ... async def click_submit() -> ActionResult:
  ...     # Implementation here
  ...     return ActionResult(...)

#### Tools.act

```python
@observe_debug(ignore_input=True, ignore_output=True, name='act')
@time_execution_sync('--act')
async def act(self, action: ActionModel, browser_session: BrowserSession, page_extraction_llm: BaseChatModel | None = None, sensitive_data: dict[str, str | dict[str, str]] | None = None, available_file_paths: list[str] | None = None, file_system: FileSystem | None = None) -> ActionResult
```

Execute an action selected by the agent.

This is the main execution method that processes ActionModel instances from
the agent and routes them to the appropriate registered action handler.

Internally, this method:
1. Extracts the action name and parameters from the ActionModel
2. Creates observability spans if Laminar is available
3. Calls registry.execute_action() with all necessary injected parameters
4. Handles exceptions and formats them for the agent
5. Returns an ActionResult with the action's output or error

**Arguments**:

- `action` - ActionModel containing the action name and parameters to execute.
  Only one action field should be set (non-None) at a time.
- `browser_session` - The active browser session for browser control operations.
- `page_extraction_llm` - Optional LLM for content extraction tasks. Used by
  extract_structured_data action to analyze page content.
- `sensitive_data` - Optional dictionary of sensitive data for replacement.
  Keys are placeholder names, values are the actual sensitive data.
- `available_file_paths` - Optional list of file paths available for upload.
  Used by upload_file_to_element to validate file availability.
- `file_system` - Optional FileSystem instance for file operations.
  Used by write_file, read_file, and related actions.

**Returns**:

- `ActionResult` - Result of the action execution containing:
  - extracted_content: Main output from the action
  - error: Error message if action failed
  - is_done: Whether the task is complete (for done action)
  - success: Whether the action succeeded
  - metadata: Additional data from the action

**Raises**:

- `ValueError` - If the action result type is invalid (not str, ActionResult, or None)

**Example**:

  >>> tools = Tools()
  >>> action = ActionModel(click_element_by_index=ClickElementAction(index=5))
  >>> result = await tools.act(
  ...     action=action,
  ...     browser_session=browser_session
  ... )
  >>> print(result.extracted_content)  # "Clicked element with index 5"

### extract_llm_error_message

```python
def extract_llm_error_message(error: Exception) -> str
```

Extract the clean error message from an exception that may contain <llm_error_msg> tags.

If the tags are found, returns the content between them.
Otherwise, returns the original error string.

This utility function is useful when handling errors from browser actions,
as the framework wraps error messages in <llm_error_msg> tags to provide
clean, LLM-friendly error messages.

**Arguments**:

- `error` - The exception to extract the message from.

**Returns**:

  The extracted clean error message, or the original error string if no tags found.

**Example**:

  >>> try:
  ...     result = await tools.act(action, browser_session)
  ... except Exception as e:
  ...     clean_msg = extract_llm_error_message(e)
  ...     return ActionResult(error=f"Action failed: {clean_msg}")


## browser_use.sync.auth


## browser_use.sync


## browser_use.sync.service


## browser_use.config

### OldConfig

```python
class OldConfig
```

Original lazy-loading configuration class for environment variables.

#### OldConfig.BROWSER_USE_CONFIG_DIR

```python
@property
def BROWSER_USE_CONFIG_DIR(self) -> Path
```

The configuration directory for browser-use.

Returns the path to browser-use's configuration directory where config files,
profiles, and extensions are stored. Defaults to ~/.config/browseruse but can
be overridden with the BROWSER_USE_CONFIG_DIR environment variable.

**Returns**:

- `Path` - The absolute path to the browser-use configuration directory.

**Example**:

  >>> from browser_use import CONFIG
  >>> config_dir = CONFIG.BROWSER_USE_CONFIG_DIR
  >>> print(config_dir)
  /home/user/.config/browseruse

### FlatEnvConfig

```python
class FlatEnvConfig(BaseSettings)
```

All environment variables in a flat namespace.

Configuration for browser-use via environment variables. All settings can be
configured through environment variables or a .env file.

Environment Variables:
	Logging & Debugging:
		BROWSER_USE_LOGGING_LEVEL: Log level (debug, info, warning, error) - default: info
		CDP_LOGGING_LEVEL: Chrome DevTools Protocol log level - default: WARNING

	Telemetry & Privacy:
		ANONYMIZED_TELEMETRY: Enable anonymous usage stats (true/false) - default: true
			Set to false to disable all telemetry collection
		BROWSER_USE_CLOUD_SYNC: Enable cloud sync of sessions - default: same as ANONYMIZED_TELEMETRY
		BROWSER_USE_CLOUD_API_URL: Cloud API endpoint - default: https://api.browser-use.com

	LLM API Keys:
		OPENAI_API_KEY: OpenAI API key for GPT models
		ANTHROPIC_API_KEY: Anthropic API key for Claude models
		GOOGLE_API_KEY: Google API key for Gemini models (formerly GEMINI_API_KEY)
		DEEPSEEK_API_KEY: DeepSeek API key
		GROK_API_KEY: Grok/xAI API key
		NOVITA_API_KEY: Novita AI API key

	Azure OpenAI:
		AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
		AZURE_OPENAI_KEY: Azure OpenAI API key
		AZURE_OPENAI_API_VERSION: API version (e.g., "2024-02-15-preview")
		AZURE_OPENAI_DEPLOYMENT: Deployment name

	AWS Bedrock:
		AWS_ACCESS_KEY_ID: AWS access key
		AWS_SECRET_ACCESS_KEY: AWS secret key
		AWS_REGION: AWS region (e.g., "us-east-1")
		AWS_BEDROCK_MODEL_ID: Model ID for Bedrock

	Configuration Paths:
		BROWSER_USE_CONFIG_DIR: Config directory - default: ~/.config/browseruse
		BROWSER_USE_CONFIG_PATH: Config file path - default: {CONFIG_DIR}/config.json
		XDG_CONFIG_HOME: XDG config base - default: ~/.config
		XDG_CACHE_HOME: XDG cache base - default: ~/.cache

	Runtime Settings:
		IN_DOCKER: Running in Docker container (auto-detected)
		IS_IN_EVALS: Running evaluation tests
		SKIP_LLM_API_KEY_VERIFICATION: Skip API key validation
		WIN_FONT_DIR: Windows font directory - default: C:\Windows\Fonts

Example .env file:
	```
	BROWSER_USE_LOGGING_LEVEL=debug
	ANONYMIZED_TELEMETRY=false
	OPENAI_API_KEY=sk-...
	ANTHROPIC_API_KEY=sk-ant-...
	```

Usage:
	>>> from browser_use.config import CONFIG
	>>> print(CONFIG.BROWSER_USE_LOGGING_LEVEL)
	info
	>>> print(CONFIG.OPENAI_API_KEY)
	sk-...

### Config

```python
class Config
```

Backward-compatible configuration class that merges all config sources.

Re-reads environment variables on every access to maintain compatibility.


## browser_use.browser.events

### ElementSelectedEvent

```python
class ElementSelectedEvent(BaseEvent[T_EventResultType])
```

Base event for element-specific operations.

Base class for all events that operate on a specific DOM element.
Provides common functionality for element selection and validation.

**Attributes**:

- `node` - The DOM element to operate on, containing all element metadata
  including element_index, node_id, attributes, position, etc.

**Example**:

  >>> # Usually not used directly, but as a base class:
  >>> class MyCustomElementEvent(ElementSelectedEvent):
  ...     custom_param: str

### NavigateToUrlEvent

```python
class NavigateToUrlEvent(BaseEvent[None])
```

Navigate to a specific URL.

Dispatch this event to navigate the browser to a new URL, either in the
current tab or a new tab.

**Attributes**:

- `url` - The URL to navigate to (must be a valid URL with protocol)
- `wait_until` - When to consider navigation complete:
  - 'load': Wait for the load event (default)
  - 'domcontentloaded': Wait for DOMContentLoaded
  - 'networkidle': Wait for network to be idle
  - 'commit': Wait for navigation to commit
- `timeout_ms` - Optional navigation timeout in milliseconds
- `new_tab` - If True, opens URL in new tab; if False, uses current tab

**Example**:

  >>> # Navigate in current tab
  >>> event = browser_session.event_bus.dispatch(
  ...     NavigateToUrlEvent(url="https://example.com")
  ... )
  >>> await event
  >>>
  >>> # Open in new tab
  >>> event = browser_session.event_bus.dispatch(
  ...     NavigateToUrlEvent(url="https://example.com", new_tab=True)
  ... )
  >>> await event

### ClickElementEvent

```python
class ClickElementEvent(ElementSelectedEvent[dict[str, Any] | None])
```

Click an element in the page.

Dispatch this event to click on a DOM element. This is the primary way to
click elements in browser-use 0.7, replacing Playwright's element.click().
Supports various click types and modifiers like Ctrl+Click for opening links in new tabs.

**Attributes**:

- `node` - The DOM element to click on (EnhancedDOMTreeNode from get_element_by_index)
- `button` - Mouse button to use ('left', 'right', or 'middle'). Default: 'left'
- `while_holding_ctrl` - If True, holds Ctrl while clicking (opens links in new tab). Default: False

**Returns**:

  Optional dict with click metadata including coordinates and timing

**Example**:

  >>> # Common pattern: find element in state, then click it
  >>> state = await browser_session.get_browser_state_summary()
  >>> for index, elem in state.dom_state.selector_map.items():
  ...     if elem.tag_name == 'BUTTON' and 'Submit' in elem.get_all_children_text():
  ...         # Get the element and click it
  ...         node = await browser_session.get_element_by_index(index)
  ...         if node:
  ...             event = browser_session.event_bus.dispatch(
  ...                 ClickElementEvent(node=node)
  ...             )
  ...             await event
  ...             break
  >>>
  >>> # Ctrl+Click to open link in new tab
  >>> link_node = await browser_session.get_element_by_index(10)
  >>> event = browser_session.event_bus.dispatch(
  ...     ClickElementEvent(node=link_node, while_holding_ctrl=True)
  ... )
  >>> await event

### TypeTextEvent

```python
class TypeTextEvent(ElementSelectedEvent[dict | None])
```

Type text into an input element.

Dispatch this event to type text into an input field, textarea, or
any editable element. This is the primary way to type text in browser-use 0.7,
replacing Playwright's element.fill() and element.type() methods.
Can optionally clear existing content first.

**Attributes**:

- `node` - The input element to type into (EnhancedDOMTreeNode from get_element_by_index)
- `text` - The text to type
- `clear_existing` - If True, clears existing content before typing. Default: False

**Returns**:

  Optional dict with input metadata

**Example**:

  >>> # Common pattern: find input field in state, then type into it
  >>> state = await browser_session.get_browser_state_summary()
  >>> for index, elem in state.dom_state.selector_map.items():
  ...     if elem.tag_name == 'INPUT' and elem.attributes.get('name') == 'username':
  ...         # Get the element and type into it
  ...         input_node = await browser_session.get_element_by_index(index)
  ...         if input_node:
  ...             event = browser_session.event_bus.dispatch(
  ...                 TypeTextEvent(
  ...                     node=input_node,
  ...                     text="user@example.com",
  ...                     clear_existing=True
  ...                 )
  ...             )
  ...             await event
  ...             break
  >>>
  >>> # Type into a search box
  >>> search_box = await browser_session.get_element_by_index(10)
  >>> event = browser_session.event_bus.dispatch(
  ...     TypeTextEvent(
  ...         node=search_box,
  ...         text="browser automation",
  ...         clear_existing=True
  ...     )
  ... )
  >>> await event

### ScrollEvent

```python
class ScrollEvent(ElementSelectedEvent[None])
```

Scroll the page or a specific element.

Dispatch this event to scroll the page or a scrollable element in any
direction by a specified amount of pixels.

**Attributes**:

- `direction` - Scroll direction ('up', 'down', 'left', 'right')
- `amount` - Number of pixels to scroll
- `node` - Optional element to scroll; if None, scrolls the entire page

**Example**:

  >>> # Scroll page down by 500 pixels
  >>> event = browser_session.event_bus.dispatch(
  ...     ScrollEvent(direction='down', amount=500)
  ... )
  >>> await event
  >>>
  >>> # Scroll specific element
  >>> scrollable_div = await browser_session.get_element_by_index(20)
  >>> event = browser_session.event_bus.dispatch(
  ...     ScrollEvent(direction='down', amount=200, node=scrollable_div)
  ... )
  >>> await event

### SwitchTabEvent

```python
class SwitchTabEvent(BaseEvent[TargetID])
```

Switch to a different browser tab.

Dispatch this event to switch focus to a different browser tab.

**Attributes**:

- `target_id` - The target ID of the tab to switch to, or None to switch
  to the most recently opened tab

**Returns**:

  TargetID of the newly focused tab

**Example**:

  >>> # Get list of tabs
  >>> tabs = await browser_session.get_tabs()
  >>>
  >>> # Switch to specific tab
  >>> event = browser_session.event_bus.dispatch(
  ...     SwitchTabEvent(target_id=tabs[1].target_id)
  ... )
  >>> await event
  >>>
  >>> # Switch to most recent tab
  >>> event = browser_session.event_bus.dispatch(
  ...     SwitchTabEvent(target_id=None)
  ... )
  >>> await event

### CloseTabEvent

```python
class CloseTabEvent(BaseEvent[None])
```

Close a browser tab.

Dispatch this event to close a specific browser tab.

**Attributes**:

- `target_id` - The target ID of the tab to close

**Example**:

  >>> # Get list of tabs
  >>> tabs = await browser_session.get_tabs()
  >>>
  >>> # Close a specific tab
  >>> event = browser_session.event_bus.dispatch(
  ...     CloseTabEvent(target_id=tabs[2].target_id)
  ... )
  >>> await event

### ScreenshotEvent

```python
class ScreenshotEvent(BaseEvent[str])
```

Take a screenshot of the page.

Dispatch this event to capture a screenshot of the current page.

**Attributes**:

- `full_page` - If True, captures entire page; if False, captures viewport only
- `clip` - Optional dict with clipping region {x, y, width, height}

**Returns**:

  Base64-encoded PNG image string

**Example**:

  >>> # Take viewport screenshot
  >>> event = browser_session.event_bus.dispatch(
  ...     ScreenshotEvent(full_page=False)
  ... )
  >>> await event
  >>> screenshot_base64 = await event.event_result()
  >>>
  >>> # Take full page screenshot
  >>> event = browser_session.event_bus.dispatch(
  ...     ScreenshotEvent(full_page=True)
  ... )
  >>> await event
  >>> screenshot_base64 = await event.event_result()

### BrowserStateRequestEvent

```python
class BrowserStateRequestEvent(BaseEvent[BrowserStateSummary])
```

Request comprehensive browser state information.

Dispatch this event to get a complete snapshot of the current browser state,
including DOM structure, interactive elements, and optionally a screenshot.

**Attributes**:

- `include_dom` - If True, includes DOM tree and interactive elements
- `include_screenshot` - If True, includes a screenshot of the page
- `cache_clickable_elements_hashes` - If True, caches element hashes for stability
- `include_recent_events` - If True, includes recent browser events in the summary

**Returns**:

  BrowserStateSummary containing page info, DOM state, and interactive elements

**Example**:

  >>> # Get full browser state
  >>> event = browser_session.event_bus.dispatch(
  ...     BrowserStateRequestEvent(
  ...         include_dom=True,
  ...         include_screenshot=True
  ...     )
  ... )
  >>> await event
  >>> state = await event.event_result()
  >>> print(f"URL: {state.url}")
  >>> print(f"Interactive elements: {len(state.interactive_elements)}")

### GoBackEvent

```python
class GoBackEvent(BaseEvent[None])
```

Navigate back in browser history.

Dispatch this event to navigate to the previous page in browser history.

**Example**:

  >>> # Go back one page
  >>> event = browser_session.event_bus.dispatch(GoBackEvent())
  >>> await event

### GoForwardEvent

```python
class GoForwardEvent(BaseEvent[None])
```

Navigate forward in browser history.

Dispatch this event to navigate to the next page in browser history.

**Example**:

  >>> # Go forward one page
  >>> event = browser_session.event_bus.dispatch(GoForwardEvent())
  >>> await event

### RefreshEvent

```python
class RefreshEvent(BaseEvent[None])
```

Refresh/reload the current page.

Dispatch this event to reload the current page.

**Example**:

  >>> # Refresh the page
  >>> event = browser_session.event_bus.dispatch(RefreshEvent())
  >>> await event

### WaitEvent

```python
class WaitEvent(BaseEvent[None])
```

Wait for a specified number of seconds.

Dispatch this event to pause execution for a specified duration.
Useful for waiting for dynamic content to load.

**Attributes**:

- `seconds` - Number of seconds to wait (default: 3.0)
- `max_seconds` - Maximum allowed wait time (safety cap at 10.0)

**Example**:

  >>> # Wait 5 seconds
  >>> event = browser_session.event_bus.dispatch(
  ...     WaitEvent(seconds=5.0)
  ... )
  >>> await event

### SendKeysEvent

```python
class SendKeysEvent(BaseEvent[None])
```

Send keyboard keys or shortcuts.

Dispatch this event to send keyboard input or shortcuts to the page.
Supports special keys and key combinations.

**Attributes**:

- `keys` - Key string or combination (e.g., "Enter", "Control+a", "Escape")

**Example**:

  >>> # Send Enter key
  >>> event = browser_session.event_bus.dispatch(
  ...     SendKeysEvent(keys="Enter")
  ... )
  >>> await event
  >>>
  >>> # Send Ctrl+A to select all
  >>> event = browser_session.event_bus.dispatch(
  ...     SendKeysEvent(keys="Control+a")
  ... )
  >>> await event

### UploadFileEvent

```python
class UploadFileEvent(ElementSelectedEvent[None])
```

Upload a file to an element.

An event class dispatched to handle file uploads. Used when the agent
needs to upload files to file input elements on web pages.

### GetDropdownOptionsEvent

```python
class GetDropdownOptionsEvent(ElementSelectedEvent[dict[str, str]])
```

Get all options from any dropdown (native <select>, ARIA menus, or custom dropdowns).

Dispatch this event to retrieve all available options from a dropdown element.
Works with native HTML selects, ARIA menus, and custom dropdown implementations.

**Attributes**:

- `node` - The dropdown element to get options from

**Returns**:

  Dict containing:
  - 'message': Human-readable summary of options
  - 'options': JSON string of available option values
  - 'type': Type of dropdown detected

**Example**:

  >>> # Get options from a dropdown
  >>> dropdown = await browser_session.get_element_by_index(15)
  >>> event = browser_session.event_bus.dispatch(
  ...     GetDropdownOptionsEvent(node=dropdown)
  ... )
  >>> await event
  >>> options_data = await event.event_result()
  >>> print(options_data['message'])

### SelectDropdownOptionEvent

```python
class SelectDropdownOptionEvent(ElementSelectedEvent[dict[str, str]])
```

Select a dropdown option by exact text from any dropdown type.

Dispatch this event to select an option from a dropdown by its text value.
Works with native HTML selects, ARIA menus, and custom dropdown implementations.

**Attributes**:

- `node` - The dropdown element to select from
- `text` - The exact text of the option to select

**Returns**:

  Dict containing:
  - 'message': Human-readable confirmation of selection
  - 'success': Boolean indicating if selection succeeded

**Example**:

  >>> # Select an option from dropdown
  >>> dropdown = await browser_session.get_element_by_index(15)
  >>> event = browser_session.event_bus.dispatch(
  ...     SelectDropdownOptionEvent(
  ...         node=dropdown,
  ...         text="United States"
  ...     )
  ... )
  >>> await event
  >>> result = await event.event_result()

### ScrollToTextEvent

```python
class ScrollToTextEvent(BaseEvent[None])
```

Scroll to specific text on the page.

Dispatch this event to scroll the page until specific text is visible.
Raises an exception if the text is not found on the page.

**Attributes**:

- `text` - The text to scroll to
- `direction` - Search direction ('up' or 'down', default: 'down')

**Raises**:

- `Exception` - If the text is not found on the page

**Example**:

  >>> # Scroll to specific text
  >>> event = browser_session.event_bus.dispatch(
  ...     ScrollToTextEvent(text="Contact Us")
  ... )
  >>> try:
  ...     await event
  ...     print("Text found and scrolled into view")
  ... except Exception:
  ...     print("Text not found on page")

### BrowserStartEvent

```python
class BrowserStartEvent(BaseEvent)
```

Start or connect to a browser instance.

Dispatch this event to start a new browser or connect to an existing one.

**Attributes**:

- `cdp_url` - Optional Chrome DevTools Protocol URL to connect to
- `launch_options` - Dictionary of browser launch options

**Example**:

  >>> # Start a new browser
  >>> event = browser_session.event_bus.dispatch(
  ...     BrowserStartEvent(
  ...         launch_options={'headless': False}
  ...     )
  ... )
  >>> await event

### BrowserStopEvent

```python
class BrowserStopEvent(BaseEvent)
```

Stop or disconnect from browser.

Dispatch this event to stop the browser and clean up resources.

**Attributes**:

- `force` - If True, forcefully terminates the browser

**Example**:

  >>> # Gracefully stop browser
  >>> event = browser_session.event_bus.dispatch(
  ...     BrowserStopEvent(force=False)
  ... )
  >>> await event

### BrowserLaunchResult

```python
class BrowserLaunchResult(BaseModel)
```

Result of launching a browser.

Data model returned after successfully launching a browser.

**Attributes**:

- `cdp_url` - Chrome DevTools Protocol URL for connecting to the browser

### BrowserLaunchEvent

```python
class BrowserLaunchEvent(BaseEvent[BrowserLaunchResult])
```

Launch a local browser process.

Dispatch this event to launch a new browser process locally.

**Returns**:

  BrowserLaunchResult containing the CDP URL of the launched browser

**Example**:

  >>> # Launch a new browser
  >>> event = browser_session.event_bus.dispatch(
  ...     BrowserLaunchEvent()
  ... )
  >>> await event
  >>> result = await event.event_result()
  >>> print(f"Browser CDP URL: {result.cdp_url}")

### BrowserKillEvent

```python
class BrowserKillEvent(BaseEvent)
```

Kill local browser subprocess.

Dispatch this event to forcefully terminate the browser process.
This is more aggressive than BrowserStopEvent.

**Example**:

  >>> # Force kill browser process
  >>> event = browser_session.event_bus.dispatch(
  ...     BrowserKillEvent()
  ... )
  >>> await event

### BrowserConnectedEvent

```python
class BrowserConnectedEvent(BaseEvent)
```

Browser has started/connected.

This event is emitted when a browser successfully connects.
Typically used for monitoring and logging.

**Attributes**:

- `cdp_url` - The CDP URL of the connected browser

**Example**:

  >>> # Listen for browser connection
  >>> @browser_session.event_bus.on(BrowserConnectedEvent)
  >>> async def on_connected(event: BrowserConnectedEvent):
  ...     print(f"Browser connected: {event.cdp_url}")

### BrowserStoppedEvent

```python
class BrowserStoppedEvent(BaseEvent)
```

Browser has stopped/disconnected.

This event is emitted when a browser disconnects or stops.
Typically used for cleanup and error handling.

**Attributes**:

- `reason` - Optional reason for the disconnection

**Example**:

  >>> # Listen for browser disconnection
  >>> @browser_session.event_bus.on(BrowserStoppedEvent)
  >>> async def on_stopped(event: BrowserStoppedEvent):
  ...     print(f"Browser stopped: {event.reason}")

### TabCreatedEvent

```python
class TabCreatedEvent(BaseEvent)
```

A new tab was created.

This event is emitted when a new browser tab is created.

**Attributes**:

- `target_id` - The target ID of the newly created tab
- `url` - The initial URL of the new tab

**Example**:

  >>> # Listen for new tabs
  >>> @browser_session.event_bus.on(TabCreatedEvent)
  >>> async def on_tab_created(event: TabCreatedEvent):
  ...     print(f"New tab: {event.url} (ID: {event.target_id})")

### TabClosedEvent

```python
class TabClosedEvent(BaseEvent)
```

A tab was closed.

This event is emitted when a browser tab is closed.

**Attributes**:

- `target_id` - The target ID of the closed tab

**Example**:

  >>> # Listen for tab closures
  >>> @browser_session.event_bus.on(TabClosedEvent)
  >>> async def on_tab_closed(event: TabClosedEvent):
  ...     print(f"Tab closed: {event.target_id}")

### AgentFocusChangedEvent

```python
class AgentFocusChangedEvent(BaseEvent)
```

Agent focus changed to a different tab.

This event is emitted when the agent's focus switches to a different tab.
Useful for tracking which tab the agent is currently working with.

**Attributes**:

- `target_id` - The target ID of the newly focused tab
- `previous_target_id` - The target ID of the previously focused tab

**Example**:

  >>> # Track agent focus changes
  >>> @browser_session.event_bus.on(AgentFocusChangedEvent)
  >>> async def on_focus_changed(event: AgentFocusChangedEvent):
  ...     print(f"Agent switched from {event.previous_target_id} to {event.target_id}")

### NavigationStartedEvent

```python
class NavigationStartedEvent(BaseEvent)
```

Navigation to a URL has started.

This event is emitted when navigation to a new URL begins.
Used for tracking navigation lifecycle and implementing
navigation-aware functionality. In browser-use 0.7, this event
helps replace Playwright's wait_for_load_state by providing
explicit navigation lifecycle tracking.

**Attributes**:

- `target_id` - The target ID of the tab being navigated
- `url` - The URL being navigated to
- `event_timeout` - Timeout for the navigation event in seconds (default: 30.0)

**Example**:

  >>> # Listen for navigation start
  >>> @browser_session.event_bus.on(NavigationStartedEvent)
  >>> async def on_navigation_start(event: NavigationStartedEvent):
  ...     print(f"Navigating to {event.url}")
  ...     # Can track navigation timing here
  ...     self.nav_start_time = time.time()

### NavigationCompleteEvent

```python
class NavigationCompleteEvent(BaseEvent)
```

Navigation to a URL has completed.

This event is emitted when navigation to a URL completes (successfully or not).
Used for tracking navigation lifecycle, implementing post-navigation logic,
and handling navigation errors. In browser-use 0.7, this event can be used
to implement custom waiting logic for dynamic content after page navigation.

**Attributes**:

- `target_id` - The target ID of the tab that completed navigation
- `url` - The final URL after navigation (may differ from requested due to redirects)
- `status` - HTTP status code of the navigation (e.g., 200, 404, None if error)
- `error_message` - Error or timeout message if navigation had issues
- `loading_status` - Detailed loading status (e.g., network timeout info)
- `event_timeout` - Timeout for the event in seconds (default: 30.0)

**Example**:

  >>> # Listen for navigation completion
  >>> @browser_session.event_bus.on(NavigationCompleteEvent)
  >>> async def on_navigation_complete(event: NavigationCompleteEvent):
  ...     if event.error_message:
  ...         print(f"Navigation failed: {event.error_message}")
  ...     else:
  ...         print(f"Navigated to {event.url} (status: {event.status})")
  ...         # Can now poll for dynamic content
  ...         await asyncio.sleep(0.5)
  ...         state = await browser_session.get_browser_state_summary()
  ...         # Check for elements that should appear after navigation


## browser_use.browser.views

### TabInfo

```python
class TabInfo(BaseModel)
```

Represents information about a browser tab

Contains metadata about an individual browser tab including its URL,
title, and Chrome DevTools Protocol target identifiers.

**Attributes**:

- `url` - The current URL of the tab
- `title` - The page title displayed in the tab
- `target_id` - Unique identifier for the Chrome DevTools Protocol target
- `parent_target_id` - Optional ID of parent tab (for popups/iframes)

**Example**:

  >>> tabs = await browser_session.get_tabs()
  >>> for tab in tabs:
  ...     print(f'{tab.title}: {tab.url}')

### PageInfo

```python
class PageInfo(BaseModel)
```

Comprehensive page size and scroll information

Contains detailed information about page dimensions, viewport size,
and current scroll position. Used internally by BrowserStateSummary.

**Attributes**:

- `viewport_width` - Width of the visible viewport in pixels
- `viewport_height` - Height of the visible viewport in pixels
- `page_width` - Total width of the page content in pixels
- `page_height` - Total height of the page content in pixels
- `scroll_x` - Horizontal scroll position in pixels
- `scroll_y` - Vertical scroll position in pixels
- `pixels_above` - Pixels scrolled above the current viewport
- `pixels_below` - Pixels remaining below the current viewport
- `pixels_left` - Pixels scrolled to the left of viewport
- `pixels_right` - Pixels remaining to the right of viewport

### BrowserStateSummary

```python
class BrowserStateSummary
```

The summary of the browser's current state designed for an LLM to process.

Contains a complete snapshot of the browser state including DOM, screenshots,
and interactive elements. This object is returned by get_browser_state_summary()
and is the primary way to access page state in browser-use 0.7.

Main Fields:
dom_state: SerializedDOMState with DOM tree and selector_map
url: Current page URL
title: Page title
tabs: List of open tabs with their information
screenshot: Base64-encoded screenshot (if vision enabled)
page_info: Enhanced page metadata (scroll position, dimensions)

DOM State Fields (via dom_state):
selector_map: Dictionary mapping element indices to EnhancedDOMTreeNode objects
Key: integer index for element identification
Value: EnhancedDOMTreeNode with full element information
llm_representation(): Method to get string representation of DOM tree

Additional Fields:
pixels_above/below: Scroll position indicators
browser_errors: List of any browser errors encountered
is_pdf_viewer: Whether viewing a PDF document
recent_events: Summary of recent browser interactions

**Example**:

  >>> # Get browser state for element interaction
  >>> state = await browser_session.get_browser_state_summary()
  >>> print(f"URL: {state.url}")
  >>> print(f"Interactive elements: {len(state.dom_state.selector_map)}")
  >>>
  >>> # Find and interact with elements by index
  >>> for index, element in state.dom_state.selector_map.items():
  ...     if element.tag_name == 'INPUT':
  ...         print(f"Input [{index}]: {element.attributes.get('name', 'unnamed')}")
  ...         # Can now interact using the index
  ...         actual_element = await browser_session.get_element_by_index(index)
  ...         if actual_element:
  ...             await browser_session.event_bus.dispatch(
  ...                 TypeTextEvent(node=actual_element, text="example text")
  ...             )

### BrowserStateHistory

```python
class BrowserStateHistory
```

Browser state at a past point in time for LLM message history

Captures a snapshot of browser state at a specific moment, used for
maintaining conversation history with the LLM. Unlike BrowserStateSummary,
this stores screenshot paths rather than the actual screenshot data.

**Attributes**:

- `url` - The URL at the time of snapshot
- `title` - The page title at the time of snapshot
- `tabs` - List of TabInfo objects representing open tabs
- `interacted_element` - Elements that were interacted with (clicked, typed into, etc.)
- `screenshot_path` - Optional filesystem path to the saved screenshot

**Methods**:

- `get_screenshot()` - Load screenshot from disk and return as base64
- `to_dict()` - Convert to dictionary format for serialization

**Example**:

  >>> history = BrowserStateHistory(
  ...     url="https://example.com",
  ...     title="Example Page",
  ...     tabs=tabs_list,
  ...     interacted_element=[clicked_element]
  ... )
  >>> screenshot_b64 = history.get_screenshot()


## browser_use.browser


## browser_use.browser.watchdog_base


## browser_use.browser.session

Event-driven browser session with backwards compatibility.

### CDPSession

```python
class CDPSession(BaseModel)
```

Chrome DevTools Protocol session for browser automation.

Represents a CDP session bound to a specific browser target (tab, iframe, etc).
Provides low-level access to Chrome DevTools Protocol commands for advanced
browser control. Can use a shared or dedicated WebSocket connection.

**Attributes**:

- `cdp_client` - The CDP client instance for sending commands
- `target_id` - Unique identifier for the browser target
- `session_id` - CDP session identifier
- `title` - Page title of the target
- `url` - Current URL of the target
- `owns_cdp_client` - If True, this session manages its own CDP connection

  Key Methods:
- `for_target()` - Create a CDP session for a specific target
- `attach()` - Attach to target and enable CDP domains
- `disconnect()` - Detach from target and cleanup
- `get_tab_info()` - Get information about the tab
- `get_target_info()` - Get detailed target information

  CDP Domains:
  Sessions enable specific CDP domains for functionality:
  - Page: Navigation, screenshots, lifecycle events
  - DOM: Document manipulation and querying
  - Runtime: JavaScript execution
  - Network: Request/response interception
  - Emulation: Device and viewport emulation
  - Storage: Cookies and local storage

  Connection Modes:
  Shared WebSocket (default): Fast, uses existing connection
  Dedicated WebSocket: Isolated, prevents interference, slower

**Example**:

  >>> # Get session for current tab
  >>> session = await browser.get_or_create_cdp_session()
  >>> # Execute CDP command
  >>> result = await session.cdp_client.send.Page.captureScreenshot(session_id=session.session_id)
  >>> # Create isolated session with new WebSocket
  >>> isolated_session = await CDPSession.for_target(cdp_client, target_id, new_socket=True, cdp_url=cdp_url)

#### CDPSession.for_target

```python
@classmethod
async def for_target(cls, cdp_client: CDPClient, target_id: TargetID, new_socket: bool = False, cdp_url: str | None = None, domains: list[str] | None = None)
```

Create a CDP session for a target.

Factory method to create a CDP session for a specific browser target.
Supports both shared and dedicated WebSocket connections.

**Arguments**:

- `cdp_client` - Existing CDP client to use (or just for reference if creating own)
- `target_id` - Target ID to attach to
- `new_socket` - If True, create a dedicated WebSocket connection for this target.
  Use for isolation or when working with multiple targets simultaneously.
- `cdp_url` - CDP URL (required if new_socket is True)
- `domains` - List of CDP domains to enable. If None, enables default domains.

**Returns**:

  Attached CDPSession instance ready for use

**Example**:

  >>> # Create session with shared connection
  >>> session = await CDPSession.for_target(cdp_client, target_id)
  >>> # Create session with dedicated connection
  >>> session = await CDPSession.for_target(cdp_client, target_id, new_socket=True, cdp_url='ws://localhost:9222/devtools/browser/...')

#### CDPSession.attach

```python
async def attach(self, domains: list[str] | None = None) -> Self
```

Attach to the target and enable specified CDP domains.

Attaches the CDP session to a browser target and enables specified
Chrome DevTools Protocol domains for interaction.

**Arguments**:

- `domains` - List of CDP domains to enable. Uses default set if None.
- `Default` - ['Page', 'DOM', 'DOMSnapshot', 'Accessibility', 'Runtime', 'Inspector']

**Returns**:

  Self for method chaining.

**Example**:

  >>> # Attach with custom domains
  >>> session = await CDPSession.for_target(...)
  >>> await session.attach(['Page', 'Network', 'Runtime'])

#### CDPSession.disconnect

```python
async def disconnect(self) -> None
```

Disconnect and cleanup if this session owns its CDP client.

Disconnects the CDP session from its target and cleans up resources.
Only disconnects if this session owns its CDP client connection.

**Notes**:

  Sessions with shared CDP clients (owns_cdp_client=False) won't
  disconnect the underlying connection, preserving it for other sessions.

**Example**:

  >>> # Disconnect when done with session
  >>> await session.disconnect()

#### CDPSession.get_tab_info

```python
async def get_tab_info(self) -> TabInfo
```

Get tab information for this CDP session.

Retrieves information about the browser tab associated with this CDP session.

**Returns**:

  TabInfo containing target ID, URL, and title.

**Example**:

  >>> tab_info = await session.get_tab_info()
  >>> print(f'Tab: {tab_info.title} - {tab_info.url}')

#### CDPSession.get_target_info

```python
async def get_target_info(self) -> TargetInfo
```

Get target information from Chrome DevTools Protocol.

Retrieves detailed information about the browser target (tab, iframe, etc.)
from the Chrome DevTools Protocol.

**Returns**:

  TargetInfo dictionary containing:
  - targetId: Unique target identifier
  - type: Target type ("page", "iframe", "worker", etc.)
  - title: Page title
  - url: Current URL
  - attached: Whether CDP is attached
  - browserContextId: Browser context ID

**Example**:

  >>> info = await session.get_target_info()
  >>> print(f'Target type: {info["type"]}, URL: {info["url"]}')

### BrowserSession

```python
class BrowserSession(BaseModel)
```

Event-driven browser session with backwards compatibility.

This class provides a 2-layer architecture:
- High-level event handling for agents/tools
- Direct CDP/Playwright calls for browser operations

Supports both event-driven and imperative calling styles.

IMPORTANT: User-facing browser actions (navigate, click, type, etc.) are performed
through the Agent and Tools classes, not directly on BrowserSession. BrowserSession
provides the underlying CDP infrastructure and state management.

Event-Driven Architecture:
The BrowserSession uses an event bus for decoupled communication between
components. Browser actions (clicks, navigation, typing) are dispatched
as events, allowing for:
- Flexible interception and modification of browser behavior
- Recording and replay of user interactions
- Telemetry and debugging via event listeners
- Plugin-style extensions without modifying core code

Events flow: User Action → Event Dispatch → Event Handlers → Browser CDP → Result Event

Common events include NavigationEvent, ClickEvent, TypeEvent, ScreenshotEvent.
Custom handlers can be registered to observe or modify behavior.

Key Operational Methods:
execute_javascript(script): Execute JavaScript in page context
evaluate_js(script): Alias for execute_javascript (backwards compatibility)
get_browser_state_summary(): Get current DOM and interactive elements
get_current_page_url(): Get current page URL
get_current_page_title(): Get current page title
get_tabs(): Get list of all open tabs
get_tab_info(): Get current tab information
get_target_info(): Get target information
get_dom_element_by_index(index): Get DOM element by its index
get_element_by_index(index): Get element by index (with caching)
get_selector_map(): Get mapping of indices to DOM elements
remove_highlights(): Remove element highlights from the page
get_all_frames(): Get all frames in the page
find_frame_target(frame_id): Find frame target by ID

Session Control:
start(): Initialize browser and CDP connection
stop(): Gracefully close browser session
kill(): Force terminate browser process
reset(): Reset session state
connect(cdp_url): Connect to CDP WebSocket endpoint

CDP Access:
get_or_create_cdp_session(): Get CDP session for current target
cdp_client: Direct access to CDP client for advanced operations
cdp_client_for_target(target_id): Get CDP client for specific target
cdp_client_for_frame(frame_id): Get CDP client for specific frame
cdp_client_for_node(node): Get CDP client for DOM node's frame

Constructor Parameters:
id: Unique session identifier (auto-generated if None)
cdp_url: WebSocket URL for CDP connection (for remote browsers)
is_local: Whether browser runs locally (default: False, auto-detected)
browser_profile: Complete profile object (for advanced use)

Browser Launch Settings:
headless: Run without UI (default: True)
executable_path: Path to browser executable
user_data_dir: Directory for browser profile persistence
args: Additional browser command-line arguments
devtools: Open DevTools automatically (default: False)
downloads_path: Directory for downloaded files

Context Settings:
viewport: Browser viewport size {"width": 1280, "height": 720}
user_agent: Custom user agent string
permissions: Grant permissions like ["geolocation", "notifications"]
storage_state: Load cookies/localStorage from file or dict
proxy: Proxy settings {"server": "http://proxy:8080"}

Security & Domains:
allowed_domains: List of domains agent can navigate to
disable_security: Disable web security features (testing only)

Recording & Debugging:
record_video_dir: Directory to save session videos
traces_dir: Directory for Chrome trace files
highlight_elements: Highlight interacted elements

Timing:
minimum_wait_page_load_time: Min wait after navigation (seconds)
wait_for_network_idle_page_load_time: Wait for network idle
wait_between_actions: Delay between actions (seconds)

Browser configuration is stored in the browser_profile, session identity in direct fields:

Headless vs Headful:
headless=True: Faster, no UI, required for servers, may trigger bot detection
headless=False: Shows browser UI, better for debugging, handles CAPTCHAs better
Use headless=False during development to observe agent behavior
```python
# Direct settings (recommended for most users)
session = BrowserSession(headless=True, user_data_dir='./profile')

# Or use a profile (for advanced use cases)
session = BrowserSession(browser_profile=BrowserProfile(...))

# Access session fields directly, browser settings via profile or property
print(session.id)  # Session field

# Common operations (note: navigation/interaction happens via Agent and Tools)
await session.start()  # Start browser and CDP connection
state = await session.get_browser_state_summary()  # Get DOM state
url = await session.get_current_page_url()  # Get current URL
await session.execute_javascript('console.log("Hello")')  # Execute JS
```

#### BrowserSession.event_bus

```python
event_bus: EventBus = Field(default_factory=EventBus)
```

Event bus for browser session communication.

Central event system for coordinating between browser session, watchdogs, and agents.
Use event_bus.dispatch(event) to send events and await responses.
Use event_bus.on(EventClass) to listen for events (replaces Playwright's page.on()).

Common events for dispatching:
- NavigateToUrlEvent: Navigate to a URL
- ClickElementEvent: Click on an element
- TypeTextEvent: Type text into an input
- BrowserStateRequestEvent: Request current page state
- SwitchTabEvent: Switch browser tab focus

Common events for listening (replaces wait_for_load_state):
- NavigationStartedEvent: Navigation begins
- NavigationCompleteEvent: Navigation finishes (replaces wait_for_load_state)
- TabCreatedEvent: New tab opened
- TabClosedEvent: Tab closed

**Examples**:

  >>> # Dispatch events (perform actions)
  >>> await browser_session.event_bus.dispatch(NavigateToUrlEvent(url='https://example.com'))
  >>>
  >>> # Listen for events (replaces Playwright's page.on() and wait_for_load_state)
  >>> @browser_session.event_bus.on(NavigationCompleteEvent)
  >>> async def on_navigation_complete(event: NavigationCompleteEvent):
  ...     print(f"Page loaded: {event.url}")
  ...     # Custom post-navigation logic here
  >>>
  >>> # Get browser state
  >>> state_event = browser_session.event_bus.dispatch(BrowserStateRequestEvent())
  >>> state = await state_event.event_result()

#### BrowserSession.cdp_url

```python
@property
def cdp_url(self) -> str | None
```

CDP URL from browser profile.

Returns the Chrome DevTools Protocol WebSocket URL used to connect to
the browser. This is automatically set when connecting to a browser.

**Returns**:

  WebSocket URL string like "ws://localhost:9222/devtools/browser/..."
  or None if not connected.

**Example**:

  >>> print(browser_session.cdp_url)
  ws://localhost:9222/devtools/browser/abc123-def456

#### BrowserSession.is_local

```python
@property
def is_local(self) -> bool
```

Whether this is a local browser instance from browser profile.

Indicates if the browser is running locally (True) or remotely (False).
This affects download handling and other behaviors.

**Returns**:

  True if browser is local, False if remote.

**Example**:

  >>> if browser_session.is_local:
  ...     print('Browser is running locally')

#### BrowserSession.reset

```python
async def reset(self) -> None
```

Clear all cached CDP sessions with proper cleanup.

Resets the browser session state by clearing all cached data, CDP sessions,
and watchdogs. Useful for starting fresh without reconnecting to the browser.

Side Effects:
- Disconnects all CDP sessions
- Clears session pool
- Resets agent focus
- Clears cached browser state
- Clears downloaded files list
- Resets all watchdogs

**Example**:

  >>> # Reset session to clean state
  >>> await browser_session.reset()
  >>> # Now reconnect or start fresh
  >>> await browser_session.connect()

#### BrowserSession.start

```python
async def start(self) -> None
```

Start the browser session.

Launches the browser process and establishes CDP connection.
This method must be called before any browser operations.

If the session is already running, this is a no-op.

**Raises**:

- `Exception` - If browser fails to launch or connect.

**Example**:

  >>> session = BrowserSession()
  >>> await session.start()
  >>> # Browser is now ready for use

#### BrowserSession.kill

```python
async def kill(self) -> None
```

Kill the browser session and reset all state.

Completely terminates the browser process and cleans up all resources.
This method saves storage state before termination if configured.

After killing, the session can be restarted with start().

**Example**:

  >>> await session.kill()
  >>> # Browser is now terminated
  >>> await session.start()  # Can restart if needed

#### BrowserSession.stop

```python
async def stop(self) -> None
```

Stop the browser session without killing the browser process.

This clears event buses and cached state but keeps the browser alive.
Useful when you want to clean up resources but plan to reconnect later.

**Example**:

  >>> await browser_session.stop()
  >>> # Browser is still running but session is cleaned up
  >>> await browser_session.start()  # Can reconnect

#### BrowserSession.on_BrowserStartEvent

```python
async def on_BrowserStartEvent(self, event: BrowserStartEvent) -> dict[str, str]
```

Handle browser start request.

Handles the browser startup event, initializing the browser process
and establishing CDP connection.

**Arguments**:

- `event` - BrowserStartEvent triggering the browser start

**Returns**:

  Dict with 'cdp_url' key containing the CDP URL

**Notes**:

  This is typically called internally by the start() method.
  Direct use is for advanced scenarios only.

#### BrowserSession.on_NavigateToUrlEvent

```python
async def on_NavigateToUrlEvent(self, event: NavigateToUrlEvent) -> None
```

Handle navigation requests - core browser functionality.

Handles navigation events to load URLs in the browser. This method manages
tab creation, reuse, and navigation orchestration. It's the primary way
to navigate programmatically.

**Arguments**:

- `event` - NavigateToUrlEvent containing:
  - url: Target URL to navigate to
  - new_tab: Whether to open in new tab (default: False)

  Behavior:
  1. If new_tab=True:
  - Looks for existing about:blank tabs to reuse
  - Creates new tab if no reusable tab found
  - Switches to the target tab
  2. If new_tab=False:
  - Checks if URL is already open in another tab
  - Uses current tab for navigation
  3. Dispatches navigation lifecycle events:
  - TabCreatedEvent (if new tab)
  - SwitchTabEvent (to activate target)
  - NavigationStartedEvent
  - NavigationCompleteEvent
  - AgentFocusChangedEvent

**Notes**:

  This method doesn't wait for full page load. It returns after initiating
  navigation. Use NavigationCompleteEvent or wait mechanisms if you need
  to ensure the page is fully loaded.

**Example**:

  >>> # Navigate in current tab
  >>> await browser_session.event_bus.dispatch(NavigateToUrlEvent(url='https://example.com'))
  >>> # Open in new tab
  >>> await browser_session.event_bus.dispatch(NavigateToUrlEvent(url='https://google.com', new_tab=True))

#### BrowserSession.on_SwitchTabEvent

```python
async def on_SwitchTabEvent(self, event: SwitchTabEvent) -> TargetID
```

Handle tab switching - core browser functionality.

Switches browser focus to a specified tab. This updates the agent's focus
to the target tab and brings it to the foreground.

**Arguments**:

- `event` - SwitchTabEvent containing:
  - target_id: Chrome target ID of the tab to switch to

**Returns**:

  TargetID of the activated tab

**Raises**:

- `RuntimeError` - If browser is not connected
- `Exception` - If target activation fails

  Side Effects:
  - Updates self.agent_focus to the new tab
  - Brings the tab to foreground in the browser
  - Creates or reuses CDP session for the target

**Example**:

  >>> # Switch to a specific tab
  >>> target_id = await browser_session.event_bus.dispatch(SwitchTabEvent(target_id='ABC123...'))
  >>> # Switch back to main tab
  >>> await browser_session.event_bus.dispatch(SwitchTabEvent(target_id=main_tab_id))

#### BrowserSession.on_CloseTabEvent

```python
async def on_CloseTabEvent(self, event: CloseTabEvent) -> None
```

Handle tab closure - update focus if needed.

Closes a browser tab identified by its target ID. This method handles
the CDP communication to close the tab and dispatches appropriate events.

**Arguments**:

- `event` - CloseTabEvent containing:
  - target_id: Chrome target ID of the tab to close

  Side Effects:
  - Closes the specified browser tab
  - Dispatches TabClosedEvent for cleanup
  - May trigger focus change if closing current tab

**Example**:

  >>> # Close a specific tab
  >>> await browser_session.event_bus.dispatch(CloseTabEvent(target_id=background_tab_id))
  >>> # Close current tab (will auto-switch to another)
  >>> await browser_session.event_bus.dispatch(CloseTabEvent(target_id=browser_session.agent_focus.target_id))

#### BrowserSession.on_TabClosedEvent

```python
async def on_TabClosedEvent(self, event: TabClosedEvent) -> None
```

Handle tab closure - update focus if needed.

Handles tab closure events by managing focus transitions. When the current
tab is closed, automatically switches to another available tab.

**Arguments**:

- `event` - TabClosedEvent containing:
  - target_id: Chrome target ID of the closed tab

  Behavior:
  - If closed tab was current: Switches to most recent other tab
  - If closed tab was background: No action needed
  - If no tabs remain: May create new blank tab

**Notes**:

  This is typically triggered automatically after CloseTabEvent.
  You don't usually need to dispatch this directly.

#### BrowserSession.on_AgentFocusChangedEvent

```python
async def on_AgentFocusChangedEvent(self, event: AgentFocusChangedEvent) -> None
```

Handle agent focus change - update focus and clear cache.

Handles focus change events when the agent switches between tabs or targets.
Clears cached state to ensure fresh data for the new target.

**Arguments**:

- `event` - AgentFocusChangedEvent containing:
  - target_id: New target to focus on
  - url: URL of the new target

  Side Effects:
  - Clears cached browser state summary
  - Clears cached selector map
  - Dispatches DOM rebuild if DOM watchdog is active

**Notes**:

  This is automatically triggered by tab switches and navigation.
  Manual dispatch is rarely needed.

#### BrowserSession.on_FileDownloadedEvent

```python
async def on_FileDownloadedEvent(self, event: FileDownloadedEvent) -> None
```

Track downloaded files during this session.

Handles file download completion events, tracking downloaded files
for the session.

**Arguments**:

- `event` - FileDownloadedEvent containing:
  - path: Full path to the downloaded file
  - file_name: Name of the downloaded file

  Side Effects:
  - Adds file path to downloaded_files list if not already tracked

**Example**:

  >>> # Access downloaded files after downloads complete
  >>> for file_path in browser_session.downloaded_files:
  ...     print(f'Downloaded: {file_path}')

#### BrowserSession.on_BrowserStopEvent

```python
async def on_BrowserStopEvent(self, event: BrowserStopEvent) -> None
```

Handle browser stop request.

Handles browser shutdown events, performing cleanup based on whether
it's a forced stop or graceful shutdown.

**Arguments**:

- `event` - BrowserStopEvent containing:
  - force: If True, kills browser process. If False, respects keep_alive.

  Side Effects:
  - Saves storage state before shutdown (if configured)
  - Kills browser process if force=True or keep_alive=False
  - Resets session state
  - Dispatches BrowserStoppedEvent

**Notes**:

  If browser_profile.keep_alive=True and force=False, the browser
  process is kept running for future reconnection.

#### BrowserSession.get_or_create_cdp_session

```python
async def get_or_create_cdp_session(self, target_id: TargetID | None = None, focus: bool = True, new_socket: bool | None = None) -> CDPSession
```

Get or create a CDP session for a target.

Gets the current Chrome DevTools Protocol (CDP) session or creates a new one.
Used for low-level browser automation through CDP commands.

**Arguments**:

- `target_id` - Target ID to get session for. If None, uses current agent focus.
- `focus` - If True, switches agent focus to this target. If False, just returns session without changing focus.
- `new_socket` - If True, create a dedicated WebSocket connection. If None (default), creates new socket for new targets only.

**Returns**:

  CDPSession for the specified target.

  CDP Session Guidance:
  new_socket Trade-offs:
  - True: Dedicated connection, isolated from other sessions, slower
  - False: Shared connection, faster, may have interference
  - None (default): Auto-decides based on target type

  Focus Behavior:
  - focus=True: Makes target active, brings tab to front
  - focus=False: Get session without switching tabs
  - Important for multi-tab automation

  Safe Usage Patterns:
  >>> # For current tab operations
  >>> session = await browser.get_or_create_cdp_session()
  >>> # For background tab operations
  >>> session = await browser.get_or_create_cdp_session(target_id=background_tab_id, focus=False)
  >>> # For isolated operations requiring clean state
  >>> session = await browser.get_or_create_cdp_session(new_socket=True)

#### BrowserSession.current_target_id

```python
@property
def current_target_id(self) -> str | None
```

Get the current target ID.

Returns the Chrome target ID of the currently focused tab/target.

**Returns**:

  Target ID string or None if no target is focused

**Example**:

  >>> target_id = browser_session.current_target_id
  >>> print(f'Current target: {target_id}')

#### BrowserSession.current_session_id

```python
@property
def current_session_id(self) -> str | None
```

Get the current CDP session ID.

Returns the CDP session ID of the currently focused target.

**Returns**:

  Session ID string or None if no session is active

**Example**:

  >>> session_id = browser_session.current_session_id
  >>> # Use for direct CDP commands

#### BrowserSession.get_browser_state_summary

```python
async def get_browser_state_summary(self, cache_clickable_elements_hashes: bool = True, include_screenshot: bool = True, cached: bool = False, include_recent_events: bool = False) -> BrowserStateSummary
```

Get comprehensive browser state including DOM, screenshot, and tabs.

Retrieves a summary of the current page state, including the DOM tree,
screenshot, open tabs, and other browser information. This is the primary
method for getting page state in browser-use 0.7, replacing direct Playwright
page.wait_for_selector() calls.

**Arguments**:

- `cache_clickable_elements_hashes` - Whether to cache element hashes for
  performance optimization on static pages. Set to False when DOM
  changes frequently to ensure fresh indices. Default: True.
- `include_screenshot` - Whether to include a screenshot. Default: True.
- `cached` - Whether to use cached state if available. Default: False.
- `include_recent_events` - Whether to include recent browser events in context. Default: False.

**Returns**:

  BrowserStateSummary containing:
  - dom_state: SerializedDOMState with selector_map of interactive elements
  - url: Current page URL
  - title: Page title
  - tabs: List of open tabs
  - screenshot: Base64-encoded screenshot (if requested)
  - page_info: Enhanced page metadata

**Example**:

  >>> # Get current page state with interactive elements
  >>> state = await browser_session.get_browser_state_summary()
  >>> print(f"Found {len(state.dom_state.selector_map)} interactive elements")
  >>>
  >>> # Access elements by index for interaction
  >>> for index, element in state.dom_state.selector_map.items():
  ...     if element.tag_name == 'BUTTON':
  ...         print(f"Button [{index}]: {element.get_all_children_text()}")
  >>>
  >>> # Use with custom polling for dynamic content
  >>> import time
  >>> start_time = time.time()
  >>> while time.time() - start_time < 10:  # 10 second timeout
  ...     state = await browser_session.get_browser_state_summary(cached=False)
  ...     if any("Submit" in el.get_all_children_text() for el in state.dom_state.selector_map.values()):
  ...         break
  ...     await asyncio.sleep(0.5)

#### BrowserSession.attach_all_watchdogs

```python
async def attach_all_watchdogs(self) -> None
```

Initialize and attach all watchdogs with explicit handler registration.

Initializes and attaches all browser watchdogs that monitor and handle
various browser events and states. Watchdogs provide functionality like
security checks, download handling, DOM updates, and crash recovery.

Side Effects:
- Creates and attaches multiple watchdog instances
- Registers event handlers for each watchdog
- Sets _watchdogs_attached flag to prevent duplicates

Watchdogs Attached:
- LocalBrowserWatchdog: Browser process management
- DOMWatchdog: DOM serialization and updates
- SecurityWatchdog: Security checks and domain restrictions
- DownloadsWatchdog: File download handling
- AboutBlankWatchdog: New tab management
- DefaultActionWatchdog: Default browser actions
- ScreenshotWatchdog: Screenshot capture
- PermissionsWatchdog: Permission management
- PopupsWatchdog: Popup/dialog handling

**Notes**:

  This is typically called automatically during browser start.
  Manual calling is only needed if watchdogs were detached.

#### BrowserSession.connect

```python
async def connect(self, cdp_url: str | None = None) -> Self
```

Connect to a remote chromium-based browser via CDP using cdp-use.

Establishes a connection to an existing Chrome/Chromium browser instance
via Chrome DevTools Protocol. Can connect to local or remote browsers.

**Arguments**:

- `cdp_url` - CDP endpoint URL. Can be:
  - WebSocket URL: "ws://localhost:9222/devtools/browser/..."
  - HTTP URL: "http://localhost:9222" (will fetch WebSocket URL)
  - None: Uses URL from browser_profile.cdp_url

**Returns**:

  Self for method chaining.

**Raises**:

- `RuntimeError` - If connection fails or no CDP URL provided.

**Notes**:

  This method automatically:
  - Redirects chrome://newtab pages to about:blank
  - Sets up proxy authentication if configured
  - Attaches all necessary watchdogs
  - Focuses on an appropriate tab

**Example**:

  >>> # Connect to local Chrome with debugging enabled
  >>> session = BrowserSession()
  >>> await session.connect('http://localhost:9222')
  >>> # Connect to remote browser
  >>> await session.connect('ws://remote-host:9222/devtools/browser/abc123')

#### BrowserSession.get_tabs

```python
async def get_tabs(self) -> list[TabInfo]
```

Get information about all open tabs using CDP Target.getTargetInfo for speed.

Retrieves information about all open browser tabs including their URLs,
titles, and target IDs. This method is optimized for speed and handles
special cases like PDF viewers and Chrome internal pages.

**Returns**:

  List of TabInfo objects, each containing:
  - url: The tab's current URL
  - title: The tab's title (or special placeholder for new tabs)
  - target_id: Full Chrome target ID
  - tab_id: Short 4-character ID for display
  - parent_target_id: ID of parent tab (for popups/iframes)

**Example**:

  >>> tabs = await browser_session.get_tabs()
  >>> for tab in tabs:
  ...     print(f'[{tab.tab_id}] {tab.title}: {tab.url}')
  >>> # Find a specific tab
  >>> google_tab = next((t for t in tabs if 'google.com' in t.url), None)

#### BrowserSession.get_current_target_info

```python
async def get_current_target_info(self) -> TargetInfo | None
```

Get info about the current active target using CDP.

Retrieves detailed information about the currently focused browser target
(tab, iframe, etc). Returns None if no target is active.

**Returns**:

  TargetInfo dictionary containing:
  - targetId: Unique target identifier
  - type: Target type ("page", "iframe", "worker")
  - title: Page title
  - url: Current URL
  - attached: Whether CDP is attached
  - browserContextId: Browser context ID

**Example**:

  >>> info = await browser_session.get_current_target_info()
  >>> if info:
  ...     print(f'Current page: {info["title"]} ({info["url"]})')

#### BrowserSession.get_current_page_url

```python
async def get_current_page_url(self) -> str
```

Get the URL of the current page using CDP.

Returns the URL of the currently active tab. Returns "about:blank"
if no tab is active or the browser is not connected.

**Returns**:

  Current page URL as string.

**Example**:

  >>> url = await browser_session.get_current_page_url()
  >>> print(f'Currently on: {url}')

#### BrowserSession.get_current_page_title

```python
async def get_current_page_title(self) -> str
```

Get the title of the current page using CDP.

Returns the title of the currently active tab. Returns "Unknown page title"
if no tab is active or the title cannot be retrieved.

**Returns**:

  Current page title as string.

**Example**:

  >>> title = await browser_session.get_current_page_title()
  >>> print(f'Page title: {title}')

#### BrowserSession.get_dom_element_by_index

```python
async def get_dom_element_by_index(self, index: int) -> EnhancedDOMTreeNode | None
```

Get DOM element by index.

Retrieves a specific DOM element by its index from the cached selector map.
This is the primary method for getting elements after finding them in
get_browser_state_summary(). Used as part of the browser-use 0.7 pattern
for element interaction, replacing direct Playwright selector methods.

**Arguments**:

- `index` - The element index from the serialized DOM (from state.dom_state.selector_map)

**Returns**:

  EnhancedDOMTreeNode or None if index not found

**Example**:

  >>> # Common pattern: find element in state, then get it by index
  >>> state = await browser_session.get_browser_state_summary()
  >>>
  >>> # Find button with specific text
  >>> button_index = None
  >>> for idx, elem in state.dom_state.selector_map.items():
  ...     if elem.tag_name == 'BUTTON' and 'Submit' in elem.get_all_children_text():
  ...         button_index = idx
  ...         break
  >>>
  >>> # Get the actual element for interaction
  >>> if button_index:
  ...     element = await browser_session.get_dom_element_by_index(button_index)
  ...     if element:
  ...         # Now interact with the element
  ...         await browser_session.event_bus.dispatch(
  ...             ClickElementEvent(node=element)
  ...         )

**Notes**:

  The selector map is cached from the last get_browser_state_summary() call.
  If the DOM has changed significantly, call get_browser_state_summary() again
  to refresh the selector map before using this method.

#### BrowserSession.update_cached_selector_map

```python
def update_cached_selector_map(self, selector_map: dict[int, EnhancedDOMTreeNode]) -> None
```

Update the cached selector map with new DOM state.

Updates the internal cache of DOM elements with a new selector map.
This is typically called by the DOM watchdog after rebuilding the DOM
to keep the element indices synchronized with the current page state.

**Arguments**:

- `selector_map` - The new selector map from DOM serialization, mapping
  element indices to EnhancedDOMTreeNode objects

**Notes**:

  This is primarily used internally by watchdogs. Manual use is only
  needed when implementing custom DOM handling.

**Example**:

  >>> # Update selector map after custom DOM parsing
  >>> new_map = {1: node1, 2: node2, 3: node3}
  >>> browser_session.update_cached_selector_map(new_map)

#### BrowserSession.get_element_by_index

```python
async def get_element_by_index(self, index: int) -> EnhancedDOMTreeNode | None
```

Alias for get_dom_element_by_index for backwards compatibility.

An alias used in examples like find_and_apply_to_jobs.py. Retrieves a
DOM element by its index.

**Arguments**:

- `index` - The element index from the serialized DOM

**Returns**:

  EnhancedDOMTreeNode or None if index not found

#### BrowserSession.get_target_id_from_tab_id

```python
async def get_target_id_from_tab_id(self, tab_id: str) -> TargetID
```

Get the full-length TargetID from the truncated 4-char tab_id.

Resolves a shortened tab ID (typically last 4 characters shown in logs)
to the full Chrome target ID needed for CDP operations.

**Arguments**:

- `tab_id` - Short tab ID suffix (e.g., "a1b2" from logs showing "`a1b2`")

**Returns**:

  Full TargetID string

**Raises**:

- `ValueError` - If no target found with the given suffix

**Example**:

  >>> # Convert short ID from logs to full ID
  >>> full_id = await browser_session.get_target_id_from_tab_id('a1b2')
  >>> await browser_session.event_bus.dispatch(SwitchTabEvent(target_id=full_id))

#### BrowserSession.get_target_id_from_url

```python
async def get_target_id_from_url(self, url: str) -> TargetID
```

Get the TargetID from a URL.

Finds the Chrome target ID of a tab by its URL. Useful for switching to
or manipulating tabs when you know the URL but not the target ID.

**Arguments**:

- `url` - The URL to search for (exact or substring match)

**Returns**:

  TargetID of the first matching tab

**Raises**:

- `ValueError` - If no tab found with the given URL

**Notes**:

  - First attempts exact URL match
  - Falls back to substring matching if exact match fails
  - Only searches page-type targets (not iframes/workers)

**Example**:

  >>> # Find tab by URL and switch to it
  >>> target_id = await browser_session.get_target_id_from_url('https://example.com/dashboard')
  >>> await browser_session.event_bus.dispatch(SwitchTabEvent(target_id=target_id))

#### BrowserSession.get_most_recently_opened_target_id

```python
async def get_most_recently_opened_target_id(self) -> TargetID
```

Get the most recently opened target ID.

Returns the target ID of the most recently opened browser tab. Useful
for switching to newly created tabs or finding the latest tab.

**Returns**:

  TargetID of the most recently opened tab

**Raises**:

- `IndexError` - If no tabs are open

**Example**:

  >>> # Switch to the newest tab
  >>> newest_tab = await browser_session.get_most_recently_opened_target_id()
  >>> await browser_session.event_bus.dispatch(SwitchTabEvent(target_id=newest_tab))

#### BrowserSession.is_file_input

```python
def is_file_input(self, element: Any) -> bool
```

Check if element is a file input.

Checks if a given DOM element is a file input field, used to determine
whether file upload functionality should be used.

**Arguments**:

- `element` - The DOM element to check

**Returns**:

  True if element is a file input, False otherwise

**Example**:

  >>> element = await browser_session.get_dom_element_by_index(5)
  >>> if browser_session.is_file_input(element):
  ...     # Use file upload instead of typing text
  ...     await browser_session.upload_file(element, '/path/to/file')

#### BrowserSession.get_selector_map

```python
async def get_selector_map(self) -> dict[int, EnhancedDOMTreeNode]
```

Get the current selector map from cached state or DOM watchdog.

Retrieves the mapping of element indices to DOM nodes. This map is used
to resolve element references when interacting with the page.

**Returns**:

  Dictionary mapping element indices to EnhancedDOMTreeNode objects

**Notes**:

  The selector map is cached and updated by the DOM watchdog whenever
  the page changes. If no cached map exists, triggers a DOM rebuild.

**Example**:

  >>> # Get all interactive elements
  >>> selector_map = await browser_session.get_selector_map()
  >>> for index, element in selector_map.items():
  ...     if element.clickable:
  ...         print(f'Element {index}: {element.tag_name}')

#### BrowserSession.execute_javascript

```python
async def execute_javascript(self, script: str, return_result: bool = True) -> Any
```

Execute JavaScript code in the page context.

Executes arbitrary JavaScript code in the context of the current page.
Can be used to interact with the page, extract data, or modify the DOM
in ways not covered by standard actions.

**Arguments**:

- `script` - JavaScript code to execute. Can be a simple expression or
  a complex script. Use IIFE for multi-line scripts.
- `return_result` - If True, returns the result of the script execution.
  If False, executes without waiting for or returning a result.

**Returns**:

  The result of the JavaScript execution if return_result is True.
  Results are automatically serialized from JavaScript types to Python.

  Security Note:
  Be careful with untrusted scripts as they execute with full page
  permissions. Always validate and sanitize any user-provided code.

**Example**:

  >>> # Get page title
  >>> title = await browser_session.execute_javascript('document.title')
  >>> # Extract data from page
  >>> data = await browser_session.execute_javascript('''
  ...     Array.from(document.querySelectorAll('.item')).map(el => ({
  ...         title: el.querySelector('.title')?.textContent,
  ...         price: el.querySelector('.price')?.textContent
  ...     }))
  ... ''')
  >>> # Modify page
  >>> await browser_session.execute_javascript(
  ...     '''
  ...     document.body.style.backgroundColor = 'lightblue';
  ...     document.querySelector('`submit`-button').click();
  ... ''',
  ...     return_result=False,
  ... )

#### BrowserSession.evaluate_js

```python
async def evaluate_js(self, script: str, return_result: bool = True) -> Any
```

Execute JavaScript code in the page context (alias for execute_javascript).

Backwards compatibility alias for execute_javascript(). Provides the same
functionality with a shorter name that matches common browser automation
conventions.

**Arguments**:

- `script` - JavaScript code to execute
- `return_result` - If True, returns the result of the script execution

**Returns**:

  The result of the JavaScript execution if return_result is True

**Example**:

  >>> # Using the alias
  >>> title = await browser_session.evaluate_js('document.title')
  >>> # Equivalent to:
  >>> title = await browser_session.execute_javascript('document.title')

**See Also**:

- `execute_javascript` - The primary method this aliases

#### BrowserSession.remove_highlights

```python
async def remove_highlights(self) -> None
```

Remove highlights from the page using CDP.

Removes all visual highlight overlays from the page that were added
for debugging or element identification purposes.

**Notes**:

  This method removes:
  - Element highlight boxes with indices
  - Debug tooltips
  - Any browser-use specific visual overlays

**Example**:

  >>> # Remove all highlights after interaction
  >>> await browser_session.remove_highlights()

#### BrowserSession.downloaded_files

```python
@property
def downloaded_files(self) -> list[str]
```

Get list of files downloaded during this browser session.

Returns paths to all files that have been downloaded during the current
browser session. Files are automatically tracked when downloads complete.

**Returns**:

  List of absolute file paths to downloaded files.

**Example**:

  >>> # After downloading files
  >>> files = browser_session.downloaded_files
  >>> for file_path in files:
  ...     print(f'Downloaded: {file_path}')

#### BrowserSession.get_all_frames

```python
async def get_all_frames(self) -> tuple[dict[str, dict], dict[str, str]]
```

Get a complete frame hierarchy from all browser targets.

Retrieves information about all frames (main pages and iframes) across
all browser tabs. This includes cross-origin iframes if enabled in the
browser profile.

**Returns**:

  Tuple of (all_frames, target_sessions) where:
  - all_frames: dict mapping frame_id -> frame info dict containing:
  - frameId: Unique frame identifier
  - parentFrameId: Parent frame ID (if iframe)
  - url: Frame URL
  - targetId: Chrome target ID
  - isCrossOrigin: Whether frame is cross-origin
  - name: Frame name attribute
  - domainAndRegistry: Domain info
  - target_sessions: dict mapping target_id -> session_id for active sessions

**Notes**:

  Cross-origin iframe support must be enabled in BrowserProfile
  (cross_origin_iframes=True) to include out-of-process iframes.

**Example**:

  >>> frames, sessions = await browser_session.get_all_frames()
  >>> for frame_id, frame_info in frames.items():
  ...     print(f'Frame {frame_id}: {frame_info["url"]}')
  ...     if frame_info.get('isCrossOrigin'):
  ...         print('  (cross-origin iframe)')

#### BrowserSession.find_frame_target

```python
async def find_frame_target(self, frame_id: str, all_frames: dict[str, dict] | None = None) -> dict | None
```

Find the frame info for a specific frame ID.

Locates frame information in the browser's frame hierarchy. Useful for
working with iframes and cross-origin content.

**Arguments**:

- `frame_id` - The frame ID to search for
- `all_frames` - Optional pre-built frame hierarchy. If None, will call get_all_frames()

**Returns**:

  Frame info dict if found containing frameId, url, targetId, etc. None otherwise

**Example**:

  >>> # Find frame info for a specific frame
  >>> frame_info = await browser_session.find_frame_target('frame123')
  >>> if frame_info:
  ...     print(f'Frame URL: {frame_info["url"]}')

#### BrowserSession.cdp_client_for_target

```python
async def cdp_client_for_target(self, target_id: TargetID) -> CDPSession
```

Get CDP session for a specific target.

Retrieves or creates a CDP session for a specific browser target (tab,
iframe, worker, etc.). This provides low-level CDP access to that target.

**Arguments**:

- `target_id` - The target identifier.

**Returns**:

  CDP session for the target.

**Example**:

  >>> # Get CDP session for a specific tab
  >>> session = await browser_session.cdp_client_for_target(tab_id)
  >>> # Use the session for CDP commands
  >>> await session.cdp_client.send.Page.reload(session_id=session.session_id)

#### BrowserSession.cdp_client_for_frame

```python
async def cdp_client_for_frame(self, frame_id: str) -> CDPSession
```

Get a CDP client attached to the target containing the specified frame.

Builds a unified frame hierarchy from all targets to find the correct target
for any frame, including OOPIFs (Out-of-Process iframes). Essential for
working with cross-origin iframes.

**Arguments**:

- `frame_id` - The frame ID to search for

**Returns**:

  CDP session attached to the target containing the frame

**Raises**:

- `ValueError` - If the frame is not found in any target

**Notes**:

  If cross-origin iframe support is disabled in browser profile,
  returns the main session for all frames.

**Example**:

  >>> # Get CDP session for iframe operations
  >>> frame_session = await browser_session.cdp_client_for_frame('frame456')
  >>> # Execute JavaScript in that frame
  >>> await frame_session.cdp_client.send.Runtime.evaluate(params={'expression': 'document.title'}, session_id=frame_session.session_id)

#### BrowserSession.cdp_client_for_node

```python
async def cdp_client_for_node(self, node: EnhancedDOMTreeNode) -> CDPSession
```

Get CDP client for a specific DOM node based on its frame.

Returns the appropriate CDP session for interacting with a DOM node,
accounting for whether the node is in an iframe or the main frame.

**Arguments**:

- `node` - The DOM node to get a CDP session for

**Returns**:

  CDP session that can interact with the node's frame

**Notes**:

  Automatically handles cross-origin iframe nodes if enabled in profile.
  Falls back to main session if frame-specific session unavailable.

**Example**:

  >>> # Get element and interact with it in its frame
  >>> element = await browser_session.get_dom_element_by_index(10)
  >>> session = await browser_session.cdp_client_for_node(element)
  >>> # Now use session for CDP commands on that node


## browser_use.browser.watchdogs.popups_watchdog


## browser_use.browser.watchdogs.storage_state_watchdog


## browser_use.browser.watchdogs.aboutblank_watchdog


## browser_use.browser.watchdogs.local_browser_watchdog


## browser_use.browser.watchdogs.default_action_watchdog


## browser_use.browser.watchdogs


## browser_use.browser.watchdogs.permissions_watchdog


## browser_use.browser.watchdogs.dom_watchdog


## browser_use.browser.watchdogs.downloads_watchdog


## browser_use.browser.watchdogs.security_watchdog


## browser_use.browser.watchdogs.screenshot_watchdog


## browser_use.browser.watchdogs.crash_watchdog


## browser_use.browser.profile

### ViewportSize

```python
class ViewportSize(BaseModel)
```

Browser viewport size configuration.

A class to define browser viewport dimensions, used to set the
window size for browser sessions. Supports dictionary-like access
via __getitem__ and __setitem__ methods for backwards compatibility.

**Example**:

  >>> viewport = ViewportSize(width=1920, height=1080)
  >>> viewport.width
  1920
  >>> viewport['height']
  1080
  >>> viewport['width'] = 1280

### RecordHarContent

```python
class RecordHarContent(str, Enum)
```

HAR content recording modes for network capture.

Controls how HTTP Archive (HAR) content is recorded during browser sessions.
Used with BrowserProfile.record_har_content parameter.

Values:
OMIT: Skip recording response content (only metadata)
EMBED: Include response content directly in HAR file
ATTACH: Save response content as separate files

**Example**:

  >>> profile = BrowserProfile(
  ...     record_har_path="./network.har",
  ...     record_har_content=RecordHarContent.EMBED
  ... )

### RecordHarMode

```python
class RecordHarMode(str, Enum)
```

HAR recording modes for network capture detail level.

Controls the level of detail in HTTP Archive (HAR) recordings.
Used with BrowserProfile.record_har_mode parameter.

Values:
FULL: Record all network activity with full details
MINIMAL: Record only essential network information

**Example**:

  >>> profile = BrowserProfile(
  ...     record_har_path="./network.har",
  ...     record_har_mode=RecordHarMode.FULL
  ... )

### BrowserChannel

```python
class BrowserChannel(str, Enum)
```

Browser release channels for selecting specific browser versions.

Enumeration of supported browser channels for Chromium-based browsers.
Used with BrowserProfile.channel to specify which browser version to launch.

Values:
CHROMIUM: Open-source Chromium browser (default)
CHROME: Google Chrome stable release
CHROME_BETA: Google Chrome beta channel
CHROME_DEV: Google Chrome dev channel
CHROME_CANARY: Google Chrome canary (nightly) channel
MSEDGE: Microsoft Edge stable release
MSEDGE_BETA: Microsoft Edge beta channel
MSEDGE_DEV: Microsoft Edge dev channel
MSEDGE_CANARY: Microsoft Edge canary channel

**Example**:

  >>> from browser_use.browser import BrowserProfile, BrowserChannel
  >>> profile = BrowserProfile(channel=BrowserChannel.CHROME)
  >>> # Or use string directly
  >>> profile = BrowserProfile(channel="chrome-beta")

### ProxySettings

```python
class ProxySettings(BaseModel)
```

Typed proxy settings for Chromium traffic.

- server: Full proxy URL, e.g. "http://host:8080" or "socks5://host:1080"
- bypass: Comma-separated hosts to bypass (e.g. "localhost,127.0.0.1,*.internal")
- username/password: Optional credentials for authenticated proxies

### BrowserProfile

```python
class BrowserProfile(BrowserConnectArgs, BrowserLaunchPersistentContextArgs, BrowserLaunchArgs, BrowserNewContextArgs)
```

Browser configuration profile for controlling browser behavior.

A BrowserProfile is a static template collection of kwargs that can be passed to:
- BrowserType.launch(**BrowserLaunchArgs)
- BrowserType.connect(**BrowserConnectArgs)
- BrowserType.connect_over_cdp(**BrowserConnectArgs)
- BrowserType.launch_persistent_context(**BrowserLaunchPersistentContextArgs)
- BrowserContext.new_context(**BrowserNewContextArgs)
- BrowserSession(**BrowserProfile)

Key Configuration Categories:

Connection Settings:
cdp_url: WebSocket URL for remote browser connection
is_local: Whether browser runs locally (affects download behavior)

Browser Launch:
headless: Run without UI (True for servers, False for debugging)
executable_path: Custom browser binary path
user_data_dir: Directory for persistent browser profile
args: Additional command-line arguments
devtools: Auto-open DevTools panel

Context Settings:
viewport: Browser viewport size (default: 1280x720)
user_agent: Custom user agent string
storage_state: Load cookies/localStorage from file/dict
permissions: Grant permissions ["geolocation", "notifications"]

Security & Navigation:
allowed_domains: Restrict navigation to specific domains
Pattern matching: Wildcards (*) match subdomains
Examples: ["*.google.com", "https://example.com", "chrome-extension://*"]
Enforcement: Checked at tool execution time before navigation
Actions blocked if URL doesn't match any allowed pattern
Default: None (all domains allowed - use with caution!)
disable_security: Disable web security (testing only - allows CORS bypass)
proxy: ProxySettings(server="http://proxy:8080", username="user", password="pass")

Extensions:
enable_default_extensions: Auto-install ad blocker, cookie handler, URL cleaner

Recording & Debugging:
record_video_dir: Save session videos
traces_dir: Chrome trace files for performance analysis
highlight_elements: Visual highlight of interacted elements

Timing Controls (replaces Playwright's wait_for_load_state):
minimum_wait_page_load_time: Minimum wait after any navigation or page change (default: 0.25s)
Ensures basic page resources are loaded before interaction
Similar to wait_for_load_state("load") in Playwright
Increase for slower sites or complex SPAs
wait_for_network_idle_page_load_time: Wait for network to be idle (default: 0.5s)
Waits until no network requests for this duration
Replaces wait_for_load_state("networkidle") in Playwright
Critical for AJAX-heavy sites and dynamic content
Set to 2-3 seconds for complex SPAs or e-commerce sites
wait_between_actions: Delay between browser actions (default: 0.5s)
Adds stability by preventing rapid-fire interactions
Helps avoid race conditions in dynamic UIs

Advanced:
cross_origin_iframes: Process out-of-process iframes (slower)
auto_download_pdfs: Download PDFs instead of viewing
deterministic_rendering: Stable rendering for testing
profile_directory: Chrome profile folder name ("Default")

Migration from OldConfig:
BrowserProfile replaces the deprecated OldConfig class.
Use BrowserProfile for all new code.

Common Usage Examples:
>>> # Basic headless automation
>>> profile = BrowserProfile(headless=True)
>>>
>>> # Interactive debugging with custom window size
>>> profile = BrowserProfile(
...     headless=False,
...     window_size=ViewportSize(width=1920, height=1080),
...     keep_alive=True,  # Keep browser open after agent.run()
...     devtools=True
... )
>>>
>>> # Security testing with disabled protections
>>> profile = BrowserProfile(
...     disable_security=True,  # Allows CORS bypass for testing
...     allowed_domains=["*.example.com"],
...     user_data_dir="./browser_data"
... )
>>>
>>> # Full-featured session with recording
>>> profile = BrowserProfile(
...     headless=False,
...     user_data_dir="./browser_data",
...     allowed_domains=["*.example.com"],
...     proxy=ProxySettings(server="http://proxy.com:8080"),
...     record_video_dir="./recordings",
...     enable_default_extensions=True,
...     keep_alive=True
... )
>>> session = BrowserSession(browser_profile=profile)


## browser_use.observability

### observe

```python
def observe(name: str | None = None, ignore_input: bool = False, ignore_output: bool = False, metadata: dict[str, Any] | None = None, span_type: Literal['DEFAULT', 'LLM', 'TOOL'] = 'DEFAULT', **kwargs: Any) -> Callable[[F], F]
```

Observability decorator that traces function execution when lmnr is available.

This decorator will use lmnr's observe decorator if lmnr is installed,
otherwise it will be a no-op that accepts the same parameters.

**Arguments**:

- `name` - Name of the span/trace
- `ignore_input` - Whether to ignore function input parameters in tracing
- `ignore_output` - Whether to ignore function output in tracing
- `metadata` - Additional metadata to attach to the span
- `span_type` - Type of span for categorization (DEFAULT, LLM, or TOOL)
- `**kwargs` - Additional parameters passed to lmnr observe

**Returns**:

  Decorated function that may be traced depending on lmnr availability

**Example**:

  @observe(name="my_function", metadata={"version": "1.0"})
  def my_function(param1, param2):
  return param1 + param2

### get_observability_status

```python
def get_observability_status() -> dict[str, bool]
```

Get the current status of observability features.

Returns a dictionary containing the status of various observability components,
useful for debugging and understanding the current observability configuration.

**Returns**:

  Dict containing:
  - lmnr_available: Whether lmnr package is installed and available
  - debug_mode: Whether debug mode is currently enabled
  - observe_active: Whether the observe decorator is actively tracing
  - observe_debug_active: Whether observe_debug is actively tracing

**Example**:

  status = get_observability_status()
  if status['observe_active']:
  print("Observability is active")


## browser_use.mcp.__main__


## browser_use.mcp.controller


## browser_use.mcp


## browser_use.mcp.server

### BrowserUseServer

```python
class BrowserUseServer
```

MCP Server for browser-use capabilities.

Provides browser automation capabilities as an MCP (Model Context Protocol) server.
This allows external MCP clients (like Claude Desktop) to use browser-use for
web automation tasks.

Module Overview:
This server exposes browser-use functionality through MCP tools, allowing
clients to:
- Control browser navigation and interactions
- Extract content from web pages
- Run autonomous AI agents for complex tasks
- Access file system operations

Exposed MCP Tool Names and Signatures:
browser_navigate(url, new_tab?): Navigate to URL
browser_go_back(): Navigate back in history
browser_go_forward(): Navigate forward in history
browser_reload(): Reload current page
browser_click(index, new_tab?): Click element by index
browser_type(text, index?): Type text in input field
browser_press(key): Press keyboard key
browser_scroll(direction?, amount?): Scroll page
browser_screenshot(): Take screenshot (base64 PNG)
browser_get_state(): Get current page state and DOM
browser_extract_content(): Extract text content
browser_extract_markdown(): Extract as markdown
browser_select_option(index, value?): Select dropdown option
browser_close_tab(): Close current tab
browser_new_tab(): Open new tab
browser_switch_tab(index): Switch to tab by index
browser_get_tabs(): List all open tabs
browser_run_task(task, url?): Run autonomous agent task
browser_start(): Start browser session
browser_stop(): Stop browser session
browser_cleanup(): Clean up all resources

Environment Variables:
BROWSER_USE_HEADLESS: Run browser in headless mode (default: true for MCP)
BROWSER_USE_KEEP_ALIVE: Keep browser alive between requests (default: true)
BROWSER_USE_DOWNLOADS_PATH: Path for downloaded files
OPENAI_API_KEY or ANTHROPIC_API_KEY: For agent tasks

Security Notes:
- Browser runs with full system access
- Only expose to trusted MCP clients
- Consider using sandboxed environment
- Set allowed_domains in browser profile for restrictions

Example Configuration (Claude Desktop):
{
"mcpServers": {
"browser-use": {
"command": "uvx",
"args": ["browser-use", "--mcp"],
"env": {
"OPENAI_API_KEY": "sk-...",
"BROWSER_USE_HEADLESS": "true"
}
}
}
}

### main

```python
async def main(http: bool = False, port: int = 3000, json_response: bool = False)
```

Main entry point for the MCP server.

Starts the browser-use MCP server in either stdio or HTTP mode.

**Arguments**:

- `http` - If True, run in HTTP mode instead of stdio (default: False)
- `port` - Port for HTTP server (default: 3000, only used if http=True)
- `json_response` - Use JSON responses instead of SSE in HTTP mode (default: False)

**Example**:

  >>> # Run as stdio server (for MCP clients)
  >>> await main()
  >>>
  >>> # Run as HTTP server for testing
  >>> await main(http=True, port=8080)


## browser_use.mcp.client

### MCPClient

```python
class MCPClient
```

Client for connecting to MCP servers and exposing their tools as browser-use actions.

Enables integration with Model Context Protocol (MCP) servers, allowing browser-use
agents to leverage external tools and services via MCP. MCP tools are dynamically
discovered and registered as browser-use actions.

MCP Client vs Server Distinction:
MCPClient (this class): CONSUMES tools from external MCP servers
- Connects to MCP server processes (filesystem, playwright, etc.)
- Discovers available tools from the server
- Registers them as browser-use actions
- Browser-use acts as a CLIENT to external tool providers

MCP Server (separate): Browser-use can also PROVIDE tools via MCP
- Exposes browser-use capabilities to other MCP clients
- Allows external apps to use browser automation
- Run via: browser-use mcp-server
- Browser-use acts as a SERVER providing tools

Tool Schema Mapping:
MCP tools are automatically converted to browser-use actions:
- Parameters mapped to Pydantic models
- Results converted to ActionResult
- Streaming/binary outputs handled transparently

Security Considerations:
- Only connect to trusted MCP servers
- Tools execute with full system access
- Use tool_filter to restrict available tools
- Validate server responses for untrusted sources

**Example**:

  >>> mcp = MCPClient(
  ...     server_name="filesystem",
  ...     command="npx",
  ...     args=["@modelcontextprotocol/server-filesystem"],
  ...     env={"FILESYSTEM_ROOT": "/safe/path"}
  ... )
  >>> await mcp.connect()
  >>> tools = await mcp.register_to_tools(agent.tools)

#### MCPClient.connect

```python
async def connect(self) -> None
```

Connect to the MCP server and discover available tools.

Establishes a connection with an MCP server and discovers all available tools.
Tools are cached internally and can be registered with browser-use Tools after
connection is established.

**Raises**:

- `RuntimeError` - If connection to the MCP server fails.

**Example**:

  >>> mcp_client = MCPClient(
  ...     server_name="my-server",
  ...     command="npx",
  ...     args=["@mycompany/mcp-server@latest"]
  ... )
  >>> await mcp_client.connect()

#### MCPClient.disconnect

```python
async def disconnect(self) -> None
```

Disconnect from the MCP server.

Terminates the connection to the MCP server and cleans up resources.
Unregisters all MCP tools from the browser-use Tools registry.

**Example**:

  >>> await mcp_client.disconnect()

#### MCPClient.register_to_tools

```python
async def register_to_tools(self, tools: Tools, tool_filter: list[str] | None = None, prefix: str | None = None) -> None
```

Register MCP tools as actions in the browser-use tools.

Registers all discovered MCP tools as browser-use actions, making them available
for agents to use during automation tasks. Tools can be filtered or prefixed
to avoid naming conflicts.

**Arguments**:

- `tools` - Browser-use tools to register actions to
- `tool_filter` - Optional list of tool names to register (None = all tools)
- `prefix` - Optional prefix to add to action names (e.g., "playwright_")

**Example**:

  >>> tools = Tools()
  >>> await mcp_client.register_to_tools(tools)
  >>> # Now agents can use MCP tools as actions


## browser_use.controller


## browser_use.screenshots


## browser_use.screenshots.service

### ScreenshotService

```python
class ScreenshotService
```

Screenshot storage service for persisting browser screenshots.

Provides methods to save and retrieve screenshots from disk. Screenshots are
stored as PNG files in a dedicated directory and can be accessed by their
file path or loaded back as base64-encoded strings.

This service is useful for:
- Persisting screenshots across agent steps
- Creating visual logs of browser interactions
- Saving screenshots for debugging or analysis
- Managing screenshot storage lifecycle

**Attributes**:

- `agent_directory` - Root directory for agent data
- `screenshots_dir` - Subdirectory where screenshots are stored

**Example**:

  >>> # Initialize service with agent directory
  >>> screenshot_service = ScreenshotService("./agent_output")
  >>>
  >>> # Store a screenshot
  >>> state = await browser_session.get_browser_state_summary(include_screenshot=True)
  >>> path = await screenshot_service.store_screenshot(
  ...     screenshot_b64=state.screenshot,
  ...     step_number=1
  ... )
  >>> print(f"Screenshot saved to: {path}")
  >>>
  >>> # Load screenshot back
  >>> loaded_b64 = await screenshot_service.get_screenshot(path)

#### ScreenshotService.__init__

```python
def __init__(self, agent_directory: str | Path)
```

Initialize screenshot service with storage directory.

Creates a screenshot service that saves screenshots to the specified
directory. Automatically creates a 'screenshots' subdirectory if it
doesn't exist.

**Arguments**:

- `agent_directory` - Root directory for agent data. Can be a string path
  or Path object. Screenshots will be stored in a 'screenshots'
  subdirectory within this location.

**Example**:

  >>> # Create service with string path
  >>> service = ScreenshotService("./my_agent")
  >>>
  >>> # Or with Path object
  >>> from pathlib import Path
  >>> service = ScreenshotService(Path.home() / "agent_data")

#### ScreenshotService.store_screenshot

```python
async def store_screenshot(self, screenshot_b64: str, step_number: int) -> str
```

Store a base64-encoded screenshot to disk.

Saves a screenshot as a PNG file with a filename based on the step number.
The screenshot is decoded from base64 and written to disk in the screenshots
directory.

**Arguments**:

- `screenshot_b64` - Base64-encoded PNG screenshot data, as returned by
  browser_session.get_browser_state_summary(include_screenshot=True)
  or ScreenshotEvent.
- `step_number` - The step number for naming the file (e.g., step_1.png).
  Used to organize screenshots chronologically.

**Returns**:

  Full file path to the saved screenshot as a string.

**Example**:

  >>> # Get screenshot from browser state
  >>> state = await browser_session.get_browser_state_summary(include_screenshot=True)
  >>>
  >>> # Save it to disk
  >>> path = await screenshot_service.store_screenshot(
  ...     screenshot_b64=state.screenshot,
  ...     step_number=1
  ... )
  >>> print(f"Saved: {path}")  # e.g., "./agent_output/screenshots/step_1.png"

#### ScreenshotService.get_screenshot

```python
async def get_screenshot(self, screenshot_path: str) -> str | None
```

Load a screenshot from disk and return as base64.

Reads a previously saved screenshot from disk and returns it as a
base64-encoded string. This can be used to reload screenshots for
further processing or to send to AI vision models.

**Arguments**:

- `screenshot_path` - Full file path to the screenshot file, as returned
  by store_screenshot().

**Returns**:

  Base64-encoded PNG data as a string, or None if the file doesn't exist
  or the path is empty.

**Example**:

  >>> # Load a previously saved screenshot
  >>> screenshot_b64 = await screenshot_service.get_screenshot(
  ...     "./agent_output/screenshots/step_1.png"
  ... )
  >>>
  >>> if screenshot_b64:
  ...     # Use with AI vision model
  ...     vision_input = f"data:image/png;base64,{screenshot_b64}"
