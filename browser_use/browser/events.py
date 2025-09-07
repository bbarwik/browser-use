"""Event definitions for browser communication."""

import inspect
from typing import Any, Literal

from bubus import BaseEvent
from bubus.models import T_EventResultType
from cdp_use.cdp.target import TargetID
from pydantic import BaseModel, Field, field_validator

from browser_use.browser.views import BrowserStateSummary
from browser_use.dom.views import EnhancedDOMTreeNode

# ============================================================================
# Agent/Tools -> BrowserSession Events (High-level browser actions)
# ============================================================================


class ElementSelectedEvent(BaseEvent[T_EventResultType]):
	"""Base event for element-specific operations.
	
	@public
	
	Base class for all events that operate on a specific DOM element.
	Provides common functionality for element selection and validation.
	
	Attributes:
		node: The DOM element to operate on, containing all element metadata
			including element_index, node_id, attributes, position, etc.
	
	Example:
		>>> # Usually not used directly, but as a base class:
		>>> class MyCustomElementEvent(ElementSelectedEvent):
		...     custom_param: str
	"""

	node: EnhancedDOMTreeNode

	@field_validator('node', mode='before')
	@classmethod
	def serialize_node(cls, data: EnhancedDOMTreeNode | None) -> EnhancedDOMTreeNode | None:
		"""Serialize DOM node data for event handling.
		
		Args:
			data: The DOM tree node to serialize.
			
		Returns:
			Serialized DOM tree node without circular references.
		"""
		if data is None:
			return None
		return EnhancedDOMTreeNode(
			element_index=data.element_index,
			node_id=data.node_id,
			backend_node_id=data.backend_node_id,
			session_id=data.session_id,
			frame_id=data.frame_id,
			target_id=data.target_id,
			node_type=data.node_type,
			node_name=data.node_name,
			node_value=data.node_value,
			attributes=data.attributes,
			is_scrollable=data.is_scrollable,
			is_visible=data.is_visible,
			absolute_position=data.absolute_position,
			# override the circular reference fields in EnhancedDOMTreeNode as they cant be serialized and aren't needed by event handlers
			# only used internally by the DOM service during DOM tree building process, not intended public API use
			content_document=None,
			shadow_root_type=None,
			shadow_roots=[],
			parent_node=None,
			children_nodes=[],
			ax_node=None,
			snapshot_node=None,
		)


# TODO: add page handle to events
# class PageHandle(share a base with browser.session.CDPSession?):
# 	url: str
# 	target_id: TargetID
#   @classmethod
#   def from_target_id(cls, target_id: TargetID) -> Self:
#     return cls(target_id=target_id)
#   @classmethod
#   def from_target_id(cls, target_id: TargetID) -> Self:
#     return cls(target_id=target_id)
#   @classmethod
#   def from_url(cls, url: str) -> Self:
#   @property
#   def root_frame_id(self) -> str:
#     return self.target_id
#   @property
#   def session_id(self) -> str:
#     return browser_session.get_or_create_cdp_session(self.target_id).session_id

# class PageSelectedEvent(BaseEvent[T_EventResultType]):
# 	"""An event like SwitchToTabEvent(page=PageHandle) or CloseTabEvent(page=PageHandle)"""
# 	page: PageHandle


class NavigateToUrlEvent(BaseEvent[None]):
	"""Navigate to a specific URL.
	
	@public
	
	Dispatch this event to navigate the browser to a new URL, either in the
	current tab or a new tab.
	
	Attributes:
		url: The URL to navigate to (must be a valid URL with protocol)
		wait_until: When to consider navigation complete:
			- 'load': Wait for the load event (default)
			- 'domcontentloaded': Wait for DOMContentLoaded
			- 'networkidle': Wait for network to be idle
			- 'commit': Wait for navigation to commit
		timeout_ms: Optional navigation timeout in milliseconds
		new_tab: If True, opens URL in new tab; if False, uses current tab
	
	Example:
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
	"""

	url: str
	wait_until: Literal['load', 'domcontentloaded', 'networkidle', 'commit'] = 'load'
	timeout_ms: int | None = None
	new_tab: bool = Field(
		default=False, description='Set True to leave the current tab alone and open a new tab in the foreground for the new URL'
	)
	# existing_tab: PageHandle | None = None  # TODO

	# time limits enforced by bubus, not exposed to LLM:
	event_timeout: float | None = 15.0  # seconds


class ClickElementEvent(ElementSelectedEvent[dict[str, Any] | None]):
	"""Click an element in the page.
	
	@public
	
	Dispatch this event to click on a DOM element. This is the primary way to
	click elements in browser-use 0.7, replacing Playwright's element.click().
	Supports various click types and modifiers like Ctrl+Click for opening links in new tabs.
	
	Attributes:
		node: The DOM element to click on (EnhancedDOMTreeNode from get_element_by_index)
		button: Mouse button to use ('left', 'right', or 'middle'). Default: 'left'
		while_holding_ctrl: If True, holds Ctrl while clicking (opens links in new tab). Default: False
	
	Returns:
		Optional dict with click metadata including coordinates and timing
	
	Example:
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
	"""

	node: 'EnhancedDOMTreeNode'
	button: Literal['left', 'right', 'middle'] = 'left'
	while_holding_ctrl: bool = Field(
		default=False,
		description='Set True to open any link clicked in a new tab in the background, can use switch_tab(tab_id=None) after to focus it',
	)
	# click_count: int = 1           # TODO
	# expect_download: bool = False  # moved to downloads_watchdog.py

	event_timeout: float | None = 15.0  # seconds


class TypeTextEvent(ElementSelectedEvent[dict | None]):
	"""Type text into an input element.
	
	@public
	
	Dispatch this event to type text into an input field, textarea, or
	any editable element. This is the primary way to type text in browser-use 0.7,
	replacing Playwright's element.fill() and element.type() methods.
	Can optionally clear existing content first.
	
	Attributes:
		node: The input element to type into (EnhancedDOMTreeNode from get_element_by_index)
		text: The text to type
		clear_existing: If True, clears existing content before typing. Default: False
	
	Returns:
		Optional dict with input metadata
	
	Example:
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
	"""

	node: 'EnhancedDOMTreeNode'
	text: str
	clear_existing: bool = True

	event_timeout: float | None = 15.0  # seconds


class ScrollEvent(ElementSelectedEvent[None]):
	"""Scroll the page or a specific element.
	
	@public
	
	Dispatch this event to scroll the page or a scrollable element in any
	direction by a specified amount of pixels.
	
	Attributes:
		direction: Scroll direction ('up', 'down', 'left', 'right')
		amount: Number of pixels to scroll
		node: Optional element to scroll; if None, scrolls the entire page
	
	Example:
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
	"""

	direction: Literal['up', 'down', 'left', 'right']
	amount: int  # pixels
	node: 'EnhancedDOMTreeNode | None' = None  # None means scroll page

	event_timeout: float | None = 8.0  # seconds


class SwitchTabEvent(BaseEvent[TargetID]):
	"""Switch to a different browser tab.
	
	@public
	
	Dispatch this event to switch focus to a different browser tab.
	
	Attributes:
		target_id: The target ID of the tab to switch to, or None to switch
			to the most recently opened tab
	
	Returns:
		TargetID of the newly focused tab
	
	Example:
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
	"""

	target_id: TargetID | None = Field(default=None, description='None means switch to the most recently opened tab')

	event_timeout: float | None = 10.0  # seconds


class CloseTabEvent(BaseEvent[None]):
	"""Close a browser tab.
	
	@public
	
	Dispatch this event to close a specific browser tab.
	
	Attributes:
		target_id: The target ID of the tab to close
	
	Example:
		>>> # Get list of tabs
		>>> tabs = await browser_session.get_tabs()
		>>> 
		>>> # Close a specific tab
		>>> event = browser_session.event_bus.dispatch(
		...     CloseTabEvent(target_id=tabs[2].target_id)
		... )
		>>> await event
	"""

	target_id: TargetID

	event_timeout: float | None = 10.0  # seconds


class ScreenshotEvent(BaseEvent[str]):
	"""Take a screenshot of the page.
	
	@public
	
	Dispatch this event to capture a screenshot of the current page.
	
	Attributes:
		full_page: If True, captures entire page; if False, captures viewport only
		clip: Optional dict with clipping region {x, y, width, height}
	
	Returns:
		Base64-encoded PNG image string
	
	Example:
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
	"""

	full_page: bool = False
	clip: dict[str, float] | None = None  # {x, y, width, height}

	event_timeout: float | None = 8.0  # seconds


class BrowserStateRequestEvent(BaseEvent[BrowserStateSummary]):
	"""Request comprehensive browser state information.
	
	@public
	
	Dispatch this event to get a complete snapshot of the current browser state,
	including DOM structure, interactive elements, and optionally a screenshot.
	
	Attributes:
		include_dom: If True, includes DOM tree and interactive elements
		include_screenshot: If True, includes a screenshot of the page
		cache_clickable_elements_hashes: If True, caches element hashes for stability
		include_recent_events: If True, includes recent browser events in the summary
	
	Returns:
		BrowserStateSummary containing page info, DOM state, and interactive elements
	
	Example:
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
	"""

	include_dom: bool = True
	include_screenshot: bool = True
	cache_clickable_elements_hashes: bool = True
	include_recent_events: bool = False

	event_timeout: float | None = 30.0  # seconds


# class WaitForConditionEvent(BaseEvent):
# 	"""Wait for a condition."""

# 	condition: Literal['navigation', 'selector', 'timeout', 'load_state']
# 	timeout: float = 30000
# 	selector: str | None = None
# 	state: Literal['attached', 'detached', 'visible', 'hidden'] | None = None


class GoBackEvent(BaseEvent[None]):
	"""Navigate back in browser history.
	
	@public
	
	Dispatch this event to navigate to the previous page in browser history.
	
	Example:
		>>> # Go back one page
		>>> event = browser_session.event_bus.dispatch(GoBackEvent())
		>>> await event
	"""

	event_timeout: float | None = 15.0  # seconds


class GoForwardEvent(BaseEvent[None]):
	"""Navigate forward in browser history.
	
	@public
	
	Dispatch this event to navigate to the next page in browser history.
	
	Example:
		>>> # Go forward one page
		>>> event = browser_session.event_bus.dispatch(GoForwardEvent())
		>>> await event
	"""

	event_timeout: float | None = 15.0  # seconds


class RefreshEvent(BaseEvent[None]):
	"""Refresh/reload the current page.
	
	@public
	
	Dispatch this event to reload the current page.
	
	Example:
		>>> # Refresh the page
		>>> event = browser_session.event_bus.dispatch(RefreshEvent())
		>>> await event
	"""

	event_timeout: float | None = 15.0  # seconds


class WaitEvent(BaseEvent[None]):
	"""Wait for a specified number of seconds.
	
	@public
	
	Dispatch this event to pause execution for a specified duration.
	Useful for waiting for dynamic content to load.
	
	Attributes:
		seconds: Number of seconds to wait (default: 3.0)
		max_seconds: Maximum allowed wait time (safety cap at 10.0)
	
	Example:
		>>> # Wait 5 seconds
		>>> event = browser_session.event_bus.dispatch(
		...     WaitEvent(seconds=5.0)
		... )
		>>> await event
	"""

	seconds: float = 3.0
	max_seconds: float = 10.0  # Safety cap

	event_timeout: float | None = 60.0  # seconds


class SendKeysEvent(BaseEvent[None]):
	"""Send keyboard keys or shortcuts.
	
	@public
	
	Dispatch this event to send keyboard input or shortcuts to the page.
	Supports special keys and key combinations.
	
	Attributes:
		keys: Key string or combination (e.g., "Enter", "Control+a", "Escape")
	
	Example:
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
	"""

	keys: str  # e.g., "ctrl+a", "cmd+c", "Enter"

	event_timeout: float | None = 15.0  # seconds


class UploadFileEvent(ElementSelectedEvent[None]):
	"""Upload a file to an element.
	
	@public
	
	An event class dispatched to handle file uploads. Used when the agent
	needs to upload files to file input elements on web pages.
	"""
	
	node: 'EnhancedDOMTreeNode'
	file_path: str

	event_timeout: float | None = 30.0  # seconds


class GetDropdownOptionsEvent(ElementSelectedEvent[dict[str, str]]):
	"""Get all options from any dropdown (native <select>, ARIA menus, or custom dropdowns).
	
	@public
	
	Dispatch this event to retrieve all available options from a dropdown element.
	Works with native HTML selects, ARIA menus, and custom dropdown implementations.
	
	Attributes:
		node: The dropdown element to get options from
	
	Returns:
		Dict containing:
			- 'message': Human-readable summary of options
			- 'options': JSON string of available option values
			- 'type': Type of dropdown detected
	
	Example:
		>>> # Get options from a dropdown
		>>> dropdown = await browser_session.get_element_by_index(15)
		>>> event = browser_session.event_bus.dispatch(
		...     GetDropdownOptionsEvent(node=dropdown)
		... )
		>>> await event
		>>> options_data = await event.event_result()
		>>> print(options_data['message'])
	"""

	node: 'EnhancedDOMTreeNode'

	event_timeout: float | None = (
		15.0  # some dropdowns lazy-load the list of options on first interaction, so we need to wait for them to load (e.g. table filter lists can have thousands of options)
	)


class SelectDropdownOptionEvent(ElementSelectedEvent[dict[str, str]]):
	"""Select a dropdown option by exact text from any dropdown type.
	
	@public
	
	Dispatch this event to select an option from a dropdown by its text value.
	Works with native HTML selects, ARIA menus, and custom dropdown implementations.
	
	Attributes:
		node: The dropdown element to select from
		text: The exact text of the option to select
	
	Returns:
		Dict containing:
			- 'message': Human-readable confirmation of selection
			- 'success': Boolean indicating if selection succeeded
	
	Example:
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
	"""

	node: 'EnhancedDOMTreeNode'
	text: str  # The option text to select

	event_timeout: float | None = 8.0  # seconds


class ScrollToTextEvent(BaseEvent[None]):
	"""Scroll to specific text on the page.
	
	@public
	
	Dispatch this event to scroll the page until specific text is visible.
	Raises an exception if the text is not found on the page.
	
	Attributes:
		text: The text to scroll to
		direction: Search direction ('up' or 'down', default: 'down')
	
	Raises:
		Exception: If the text is not found on the page
	
	Example:
		>>> # Scroll to specific text
		>>> event = browser_session.event_bus.dispatch(
		...     ScrollToTextEvent(text="Contact Us")
		... )
		>>> try:
		...     await event
		...     print("Text found and scrolled into view")
		... except Exception:
		...     print("Text not found on page")
	"""

	text: str
	direction: Literal['up', 'down'] = 'down'

	event_timeout: float | None = 15.0  # seconds


# ============================================================================


class BrowserStartEvent(BaseEvent):
	"""Start or connect to a browser instance.
	
	@public
	
	Dispatch this event to start a new browser or connect to an existing one.
	
	Attributes:
		cdp_url: Optional Chrome DevTools Protocol URL to connect to
		launch_options: Dictionary of browser launch options
	
	Example:
		>>> # Start a new browser
		>>> event = browser_session.event_bus.dispatch(
		...     BrowserStartEvent(
		...         launch_options={'headless': False}
		...     )
		... )
		>>> await event
	"""

	cdp_url: str | None = None
	launch_options: dict[str, Any] = Field(default_factory=dict)

	event_timeout: float | None = 30.0  # seconds


class BrowserStopEvent(BaseEvent):
	"""Stop or disconnect from browser.
	
	@public
	
	Dispatch this event to stop the browser and clean up resources.
	
	Attributes:
		force: If True, forcefully terminates the browser
	
	Example:
		>>> # Gracefully stop browser
		>>> event = browser_session.event_bus.dispatch(
		...     BrowserStopEvent(force=False)
		... )
		>>> await event
	"""

	force: bool = False

	event_timeout: float | None = 45.0  # seconds


class BrowserLaunchResult(BaseModel):
	"""Result of launching a browser.
	
	@public
	
	Data model returned after successfully launching a browser.
	
	Attributes:
		cdp_url: Chrome DevTools Protocol URL for connecting to the browser
	"""

	# TODO: add browser executable_path, pid, version, latency, user_data_dir, X11 $DISPLAY, host IP address, etc.
	cdp_url: str


class BrowserLaunchEvent(BaseEvent[BrowserLaunchResult]):
	"""Launch a local browser process.
	
	@public
	
	Dispatch this event to launch a new browser process locally.
	
	Returns:
		BrowserLaunchResult containing the CDP URL of the launched browser
	
	Example:
		>>> # Launch a new browser
		>>> event = browser_session.event_bus.dispatch(
		...     BrowserLaunchEvent()
		... )
		>>> await event
		>>> result = await event.event_result()
		>>> print(f"Browser CDP URL: {result.cdp_url}")
	"""

	# TODO: add executable_path, proxy settings, preferences, extra launch args, etc.

	event_timeout: float | None = 30.0  # seconds


class BrowserKillEvent(BaseEvent):
	"""Kill local browser subprocess.
	
	@public
	
	Dispatch this event to forcefully terminate the browser process.
	This is more aggressive than BrowserStopEvent.
	
	Example:
		>>> # Force kill browser process
		>>> event = browser_session.event_bus.dispatch(
		...     BrowserKillEvent()
		... )
		>>> await event
	"""

	event_timeout: float | None = 30.0  # seconds


# TODO: replace all Runtime.evaluate() calls with this event
# class ExecuteJavaScriptEvent(BaseEvent):
# 	"""Execute JavaScript in page context."""

# 	target_id: TargetID
# 	expression: str
# 	await_promise: bool = True

# 	event_timeout: float | None = 60.0  # seconds

# TODO: add this and use the old BrowserProfile.viewport options to set it
# class SetViewportEvent(BaseEvent):
# 	"""Set the viewport size."""

# 	width: int
# 	height: int
# 	device_scale_factor: float = 1.0

# 	event_timeout: float | None = 15.0  # seconds


# Moved to storage state
# class SetCookiesEvent(BaseEvent):
# 	"""Set browser cookies."""

# 	cookies: list[dict[str, Any]]

# 	event_timeout: float | None = (
# 		30.0  # only long to support the edge case of restoring a big localStorage / on many origins (has to O(n) visit each origin to restore)
# 	)


# class GetCookiesEvent(BaseEvent):
# 	"""Get browser cookies."""

# 	urls: list[str] | None = None

# 	event_timeout: float | None = 30.0  # seconds


# ============================================================================
# DOM-related Events
# ============================================================================


class BrowserConnectedEvent(BaseEvent):
	"""Browser has started/connected.
	
	@public
	
	This event is emitted when a browser successfully connects.
	Typically used for monitoring and logging.
	
	Attributes:
		cdp_url: The CDP URL of the connected browser
	
	Example:
		>>> # Listen for browser connection
		>>> @browser_session.event_bus.on(BrowserConnectedEvent)
		>>> async def on_connected(event: BrowserConnectedEvent):
		...     print(f"Browser connected: {event.cdp_url}")
	"""

	cdp_url: str

	event_timeout: float | None = 30.0  # seconds


class BrowserStoppedEvent(BaseEvent):
	"""Browser has stopped/disconnected.
	
	@public
	
	This event is emitted when a browser disconnects or stops.
	Typically used for cleanup and error handling.
	
	Attributes:
		reason: Optional reason for the disconnection
	
	Example:
		>>> # Listen for browser disconnection
		>>> @browser_session.event_bus.on(BrowserStoppedEvent)
		>>> async def on_stopped(event: BrowserStoppedEvent):
		...     print(f"Browser stopped: {event.reason}")
	"""

	reason: str | None = None

	event_timeout: float | None = 30.0  # seconds


class TabCreatedEvent(BaseEvent):
	"""A new tab was created.
	
	@public
	
	This event is emitted when a new browser tab is created.
	
	Attributes:
		target_id: The target ID of the newly created tab
		url: The initial URL of the new tab
	
	Example:
		>>> # Listen for new tabs
		>>> @browser_session.event_bus.on(TabCreatedEvent)
		>>> async def on_tab_created(event: TabCreatedEvent):
		...     print(f"New tab: {event.url} (ID: {event.target_id})")
	"""

	target_id: TargetID
	url: str

	event_timeout: float | None = 30.0  # seconds


class TabClosedEvent(BaseEvent):
	"""A tab was closed.
	
	@public
	
	This event is emitted when a browser tab is closed.
	
	Attributes:
		target_id: The target ID of the closed tab
	
	Example:
		>>> # Listen for tab closures
		>>> @browser_session.event_bus.on(TabClosedEvent)
		>>> async def on_tab_closed(event: TabClosedEvent):
		...     print(f"Tab closed: {event.target_id}")
	"""

	target_id: TargetID

	# TODO:
	# new_focus_target_id: int | None = None
	# new_focus_url: str | None = None

	event_timeout: float | None = 10.0  # seconds


# TODO: emit this when DOM changes significantly, inner frame navigates, form submits, history.pushState(), etc.
# class TabUpdatedEvent(BaseEvent):
# 	"""Tab information updated (URL changed, etc.)."""

# 	target_id: TargetID
# 	url: str


class AgentFocusChangedEvent(BaseEvent):
	"""Agent focus changed to a different tab.
	
	@public
	
	This event is emitted when the agent's focus switches to a different tab.
	Useful for tracking which tab the agent is currently working with.
	
	Attributes:
		target_id: The target ID of the newly focused tab
		previous_target_id: The target ID of the previously focused tab
	
	Example:
		>>> # Track agent focus changes
		>>> @browser_session.event_bus.on(AgentFocusChangedEvent)
		>>> async def on_focus_changed(event: AgentFocusChangedEvent):
		...     print(f"Agent switched from {event.previous_target_id} to {event.target_id}")
	"""

	target_id: TargetID
	url: str

	event_timeout: float | None = 10.0  # seconds


class TargetCrashedEvent(BaseEvent):
	"""A target has crashed."""

	target_id: TargetID
	error: str

	event_timeout: float | None = 10.0  # seconds


class NavigationStartedEvent(BaseEvent):
	"""Navigation to a URL has started.
	
	@public
	
	This event is emitted when navigation to a new URL begins.
	Used for tracking navigation lifecycle and implementing
	navigation-aware functionality. In browser-use 0.7, this event
	helps replace Playwright's wait_for_load_state by providing
	explicit navigation lifecycle tracking.
	
	Attributes:
		target_id: The target ID of the tab being navigated
		url: The URL being navigated to
		event_timeout: Timeout for the navigation event in seconds (default: 30.0)
	
	Example:
		>>> # Listen for navigation start
		>>> @browser_session.event_bus.on(NavigationStartedEvent)
		>>> async def on_navigation_start(event: NavigationStartedEvent):
		...     print(f"Navigating to {event.url}")
		...     # Can track navigation timing here
		...     self.nav_start_time = time.time()
	"""

	target_id: TargetID
	url: str

	event_timeout: float | None = 30.0  # seconds


class NavigationCompleteEvent(BaseEvent):
	"""Navigation to a URL has completed.
	
	@public
	
	This event is emitted when navigation to a URL completes (successfully or not).
	Used for tracking navigation lifecycle, implementing post-navigation logic,
	and handling navigation errors. In browser-use 0.7, this event can be used
	to implement custom waiting logic for dynamic content after page navigation.
	
	Attributes:
		target_id: The target ID of the tab that completed navigation
		url: The final URL after navigation (may differ from requested due to redirects)
		status: HTTP status code of the navigation (e.g., 200, 404, None if error)
		error_message: Error or timeout message if navigation had issues
		loading_status: Detailed loading status (e.g., network timeout info)
		event_timeout: Timeout for the event in seconds (default: 30.0)
	
	Example:
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
	"""

	target_id: TargetID
	url: str
	status: int | None = None
	error_message: str | None = None  # Error/timeout message if navigation had issues
	loading_status: str | None = None  # Detailed loading status (e.g., network timeout info)

	event_timeout: float | None = 30.0  # seconds


# ============================================================================
# Error Events
# ============================================================================


class BrowserErrorEvent(BaseEvent):
	"""An error occurred in the browser layer."""

	error_type: str
	message: str
	details: dict[str, Any] = Field(default_factory=dict)

	event_timeout: float | None = 30.0  # seconds


# ============================================================================
# Storage State Events
# ============================================================================


class SaveStorageStateEvent(BaseEvent):
	"""Request to save browser storage state."""

	path: str | None = None  # Optional path, uses profile default if not provided

	event_timeout: float | None = 45.0  # seconds


class StorageStateSavedEvent(BaseEvent):
	"""Notification that storage state was saved."""

	path: str
	cookies_count: int
	origins_count: int

	event_timeout: float | None = 30.0  # seconds


class LoadStorageStateEvent(BaseEvent):
	"""Request to load browser storage state."""

	path: str | None = None  # Optional path, uses profile default if not provided

	event_timeout: float | None = 45.0  # seconds


# TODO: refactor this to:
# - on_BrowserConnectedEvent() -> dispatch(LoadStorageStateEvent()) -> _copy_storage_state_from_json_to_browser(json_file, new_cdp_session) + return storage_state from handler
# - on_BrowserStopEvent() -> dispatch(SaveStorageStateEvent()) -> _copy_storage_state_from_browser_to_json(new_cdp_session, json_file)
# and get rid of StorageStateSavedEvent and StorageStateLoadedEvent, have the original events + provide handler return values for any results
class StorageStateLoadedEvent(BaseEvent):
	"""Notification that storage state was loaded."""

	path: str
	cookies_count: int
	origins_count: int

	event_timeout: float | None = 30.0  # seconds


# ============================================================================
# File Download Events
# ============================================================================


class FileDownloadedEvent(BaseEvent):
	"""A file has been downloaded."""

	url: str
	path: str
	file_name: str
	file_size: int
	file_type: str | None = None  # e.g., 'pdf', 'zip', 'docx', etc.
	mime_type: str | None = None  # e.g., 'application/pdf'
	from_cache: bool = False
	auto_download: bool = False  # Whether this was an automatic download (e.g., PDF auto-download)

	event_timeout: float | None = 30.0  # seconds


class AboutBlankDVDScreensaverShownEvent(BaseEvent):
	"""AboutBlankWatchdog has shown DVD screensaver animation on an about:blank tab."""

	target_id: TargetID
	error: str | None = None


class DialogOpenedEvent(BaseEvent):
	"""Event dispatched when a JavaScript dialog is opened and handled."""

	dialog_type: str  # 'alert', 'confirm', 'prompt', or 'beforeunload'
	message: str
	url: str
	frame_id: str
	# target_id: TargetID   # TODO: add this to avoid needing target_id_from_frame() later


# Note: Model rebuilding for forward references is handled in the importing modules
# Events with 'EnhancedDOMTreeNode' forward references (ClickElementEvent, TypeTextEvent,
# ScrollEvent, UploadFileEvent) need model_rebuild() called after imports are complete


def _check_event_names_dont_overlap():
	"""Check that event names defined in this file are valid and non-overlapping
	(naiively n^2 so it's pretty slow but ok for now, optimize when >20 events)
	"""
	event_names = {
		name.split('[')[0]
		for name in globals().keys()
		if not name.startswith('_')
		and inspect.isclass(globals()[name])
		and issubclass(globals()[name], BaseEvent)
		and name != 'BaseEvent'
	}
	for name_a in event_names:
		assert name_a.endswith('Event'), f'Event with name {name_a} does not end with "Event"'
		for name_b in event_names:
			if name_a != name_b:  # Skip self-comparison
				assert name_a not in name_b, (
					f'Event with name {name_a} is a substring of {name_b}, all events must be completely unique to avoid find-and-replace accidents'
				)


# overlapping event names are a nightmare to trace and rename later, dont do it!
# e.g. prevent ClickEvent and FailedClickEvent are terrible names because one is a substring of the other,
# must be ClickEvent and ClickFailedEvent to preserve the usefulnes of codebase grep/sed/awk as refactoring tools.
# at import time, we do a quick check that all event names defined above are valid and non-overlapping.
# this is hand written in blood by a human! not LLM slop. feel free to optimize but do not remove it without a good reason.
_check_event_names_dont_overlap()
