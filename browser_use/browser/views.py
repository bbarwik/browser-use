"""Browser views and data models for tab information and state management."""
from dataclasses import dataclass, field
from typing import Any

from bubus import BaseEvent
from cdp_use.cdp.target import TargetID
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_serializer

from browser_use.dom.views import DOMInteractedElement, SerializedDOMState

# Known placeholder image data for about:blank pages - a 4x4 white PNG
PLACEHOLDER_4PX_SCREENSHOT = (
	'iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAFElEQVR4nGP8//8/AwwwMSAB3BwAlm4DBfIlvvkAAAAASUVORK5CYII='
)


# Pydantic
class TabInfo(BaseModel):
	"""Represents information about a browser tab
	
	@public
	
	Contains metadata about an individual browser tab including its URL,
	title, and Chrome DevTools Protocol target identifiers.
	
	Attributes:
		url: The current URL of the tab
		title: The page title displayed in the tab
		target_id: Unique identifier for the Chrome DevTools Protocol target
		parent_target_id: Optional ID of parent tab (for popups/iframes)
	
	Example:
		>>> tabs = await browser_session.get_tabs()
		>>> for tab in tabs:
		...     print(f'{tab.title}: {tab.url}')
	"""

	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		populate_by_name=True,
	)

	# Original fields
	url: str
	title: str
	target_id: TargetID = Field(serialization_alias='tab_id', validation_alias=AliasChoices('tab_id', 'target_id'))
	parent_target_id: TargetID | None = Field(
		default=None, serialization_alias='parent_tab_id', validation_alias=AliasChoices('parent_tab_id', 'parent_target_id')
	)  # parent page that contains this popup or cross-origin iframe

	@field_serializer('target_id')
	def serialize_target_id(self, target_id: TargetID, _info: Any) -> str:
		"""Serialize target ID to short form for display.
		
		Args:
			target_id: The target identifier.
			_info: Serialization info (unused).
			
		Returns:
			Last 4 characters of the target ID.
		"""
		return target_id[-4:]

	@field_serializer('parent_target_id')
	def serialize_parent_target_id(self, parent_target_id: TargetID | None, _info: Any) -> str | None:
		"""Serialize parent target ID to short form for display.
		
		Args:
			parent_target_id: The parent target identifier.
			_info: Serialization info (unused).
			
		Returns:
			Last 4 characters of parent target ID, or None if no parent.
		"""
		return parent_target_id[-4:] if parent_target_id else None


class PageInfo(BaseModel):
	"""Comprehensive page size and scroll information
	
	@public
	
	Contains detailed information about page dimensions, viewport size,
	and current scroll position. Used internally by BrowserStateSummary.
	
	Attributes:
		viewport_width: Width of the visible viewport in pixels
		viewport_height: Height of the visible viewport in pixels
		page_width: Total width of the page content in pixels
		page_height: Total height of the page content in pixels
		scroll_x: Horizontal scroll position in pixels
		scroll_y: Vertical scroll position in pixels
		pixels_above: Pixels scrolled above the current viewport
		pixels_below: Pixels remaining below the current viewport
		pixels_left: Pixels scrolled to the left of viewport
		pixels_right: Pixels remaining to the right of viewport
	"""

	# Current viewport dimensions
	viewport_width: int
	viewport_height: int

	# Total page dimensions
	page_width: int
	page_height: int

	# Current scroll position
	scroll_x: int
	scroll_y: int

	# Calculated scroll information
	pixels_above: int
	pixels_below: int
	pixels_left: int
	pixels_right: int

	# Page statistics are now computed dynamically instead of stored


@dataclass
class BrowserStateSummary:
	"""The summary of the browser's current state designed for an LLM to process.
	
	@public
	
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
	
	Example:
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
	"""

	# provided by SerializedDOMState:
	dom_state: SerializedDOMState

	url: str
	title: str
	tabs: list[TabInfo]
	screenshot: str | None = field(default=None, repr=False)
	page_info: PageInfo | None = None  # Enhanced page information

	# Keep legacy fields for backward compatibility
	pixels_above: int = 0
	pixels_below: int = 0
	browser_errors: list[str] = field(default_factory=list)
	is_pdf_viewer: bool = False  # Whether the current page is a PDF viewer
	recent_events: str | None = None  # Text summary of recent browser events


@dataclass
class BrowserStateHistory:
	"""Browser state at a past point in time for LLM message history
	
	@public
	
	Captures a snapshot of browser state at a specific moment, used for
	maintaining conversation history with the LLM. Unlike BrowserStateSummary,
	this stores screenshot paths rather than the actual screenshot data.
	
	Attributes:
		url: The URL at the time of snapshot
		title: The page title at the time of snapshot
		tabs: List of TabInfo objects representing open tabs
		interacted_element: Elements that were interacted with (clicked, typed into, etc.)
		screenshot_path: Optional filesystem path to the saved screenshot
	
	Methods:
		get_screenshot(): Load screenshot from disk and return as base64
		to_dict(): Convert to dictionary format for serialization
	
	Example:
		>>> history = BrowserStateHistory(
		...     url="https://example.com",
		...     title="Example Page",
		...     tabs=tabs_list,
		...     interacted_element=[clicked_element]
		... )
		>>> screenshot_b64 = history.get_screenshot()
	"""

	url: str
	title: str
	tabs: list[TabInfo]
	interacted_element: list[DOMInteractedElement | None] | list[None]
	screenshot_path: str | None = None

	def get_screenshot(self) -> str | None:
		"""Load screenshot from disk and return as base64 string"""
		if not self.screenshot_path:
			return None

		import base64
		from pathlib import Path

		path_obj = Path(self.screenshot_path)
		if not path_obj.exists():
			return None

		try:
			with open(path_obj, 'rb') as f:
				screenshot_data = f.read()
			return base64.b64encode(screenshot_data).decode('utf-8')
		except Exception:
			return None

	def to_dict(self) -> dict[str, Any]:
		"""Convert browser state history to dictionary format.
		
		Returns:
			Dictionary representation of the browser state history.
		"""
		data = {}
		data['tabs'] = [tab.model_dump() for tab in self.tabs]
		data['screenshot_path'] = self.screenshot_path
		data['interacted_element'] = [el.to_dict() if el else None for el in self.interacted_element]
		data['url'] = self.url
		data['title'] = self.title
		return data


class BrowserError(Exception):
	"""Base class for all browser errors"""

	message: str
	details: dict[str, Any] | None = None
	while_handling_event: BaseEvent[Any] | None = None

	def __init__(self, message: str, details: dict[str, Any] | None = None, event: BaseEvent[Any] | None = None):
		self.message = message
		super().__init__(message)
		self.details = details
		self.while_handling_event = event

	def __str__(self) -> str:
		if self.details:
			return f'{self.message} ({self.details}) during: {self.while_handling_event}'
		elif self.while_handling_event:
			return f'{self.message} (while handling: {self.while_handling_event})'
		else:
			return self.message


class URLNotAllowedError(BrowserError):
	"""Error raised when a URL is not allowed"""
