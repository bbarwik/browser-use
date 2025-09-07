"""Tool action data models and views."""
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field


# Action Input Models
class SearchGoogleAction(BaseModel):
	"""Action model for performing a Google search.
	
	@public
	
	Performs a Google search with the specified query. The browser will
	navigate to Google and execute the search.
	
	Args:
		query: The search query string to execute on Google.
	
	Example:
		>>> action = SearchGoogleAction(query="browser automation python")
	"""
	query: str


class GoToUrlAction(BaseModel):
	"""Action model for navigating to a URL.
	
	@public
	
	Navigates the browser to the specified URL, either in the current tab
	or a new tab.
	
	Args:
		url: The URL to navigate to.
		new_tab: Whether to open the URL in a new tab (True) or navigate in current tab (False).
	
	Example:
		>>> # Navigate in current tab
		>>> action = GoToUrlAction(url="https://example.com")
		>>>
		>>> # Open in new tab
		>>> action = GoToUrlAction(url="https://example.com", new_tab=True)
	"""
	url: str
	new_tab: bool = False  # True to open in new tab, False to navigate in current tab


class ClickElementAction(BaseModel):
	"""Action model for clicking on a page element.
	
	@public
	
	Clicks on an element identified by its index in the DOM. Elements are
	indexed starting from 1 based on their appearance in the interactive
	elements list.
	
	Args:
		index: The index of the element to click (1-based indexing).
		while_holding_ctrl: Whether to hold Ctrl while clicking to open links in new background tabs.
	
	Example:
		>>> # Click the 5th interactive element
		>>> action = ClickElementAction(index=5)
		>>>
		>>> # Click with Ctrl held to open in new tab
		>>> action = ClickElementAction(index=3, while_holding_ctrl=True)
	"""
	index: int = Field(ge=1, description='index of the element to click')
	while_holding_ctrl: bool = Field(
		default=False, description='set True to open any resulting navigation in a new background tab, False otherwise'
	)
	# expect_download: bool = Field(default=False, description='set True if expecting a download, False otherwise')  # moved to downloads_watchdog.py
	# click_count: int = 1  # TODO


class InputTextAction(BaseModel):
	"""Action model for inputting text into a page element.
	
	@public
	
	Inputs text into a form field or text area. Can either replace existing
	text or append to it.
	
	Args:
		index: The index of the element to input text into (0 for page, 1+ for specific elements).
		text: The text content to input.
		clear_existing: Whether to clear existing text (True) or append to it (False).
	
	Example:
		>>> # Type into the 3rd input field, replacing existing text
		>>> action = InputTextAction(index=3, text="Hello World")
		>>>
		>>> # Append to existing text in field 2
		>>> action = InputTextAction(index=2, text=" additional text", clear_existing=False)
	"""
	index: int = Field(ge=0, description='index of the element to input text into, 0 is the page')
	text: str
	clear_existing: bool = Field(default=True, description='set True to clear existing text, False to append to existing text')


class DoneAction(BaseModel):
	"""Action model for marking a task as complete.
	
	@public
	
	Signals that a task or automation sequence has been completed.
	Used to indicate success/failure and provide a summary of results.
	
	Args:
		text: Description or summary of what was accomplished.
		success: Whether the task completed successfully.
		files_to_display: Optional list of file paths to display or reference.
	
	Example:
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
	"""
	text: str
	success: bool
	files_to_display: list[str] | None = []


T = TypeVar('T', bound=BaseModel)


class StructuredOutputAction(BaseModel, Generic[T]):
	"""Generic action model for returning structured data.
	
	@public
	
	Used when an agent is configured with a structured output model. This action
	replaces the standard 'done' action to return typed, validated data according
	to a Pydantic model schema.
	
	Type Parameters:
		T: The Pydantic model type for the structured output
		
	Args:
		success: Whether the operation was successful.
		data: The structured data of type T to return.
		
	Example:
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
	"""
	success: bool = True
	data: T


class SwitchTabAction(BaseModel):
	"""Action model for switching to a different browser tab.
	
	@public
	
	Switches browser focus to a different tab, identified either by URL
	pattern or by its Tab ID (last 4 characters of the target ID).
	
	Args:
		url: URL or URL substring of the tab to switch to.
		tab_id: Exact 4-character Tab ID to match instead of URL.
	
	Example:
		>>> # Switch by URL pattern
		>>> action = SwitchTabAction(url="example.com")
		>>>
		>>> # Switch by exact Tab ID
		>>> action = SwitchTabAction(tab_id="A1B2")
	"""
	url: str | None = Field(
		default=None,
		description='URL or URL substring of the tab to switch to, if not provided, the tab_id or most recently opened tab will be used',
	)
	tab_id: str | None = Field(
		default=None,
		min_length=4,
		max_length=4,
		description='exact 4 character Tab ID to match instead of URL, prefer using this if known',
	)  # last 4 chars of TargetID


class CloseTabAction(BaseModel):
	"""Action model for closing a browser tab.
	
	@public
	
	Closes a specific browser tab identified by its Tab ID.
	
	Args:
		tab_id: The 4-character Tab ID of the tab to close (last 4 chars of TargetID).
	
	Example:
		>>> # Close tab with ID "A1B2"
		>>> action = CloseTabAction(tab_id="A1B2")
	"""
	tab_id: str = Field(min_length=4, max_length=4, description='4 character Tab ID')  # last 4 chars of TargetID


class ScrollAction(BaseModel):
	"""Action model for scrolling the page or a specific element.
	
	@public
	
	Scrolls the page or a specific scrollable element by the specified amount.
	
	Args:
		down: Direction to scroll - True for down, False for up.
		num_pages: Number of pages to scroll (e.g., 0.5 = half page, 1.0 = one page).
		frame_element_index: Optional element index to find scroll container for.
	
	Example:
		>>> # Scroll down one page
		>>> action = ScrollAction(down=True, num_pages=1.0)
		>>>
		>>> # Scroll up half a page
		>>> action = ScrollAction(down=False, num_pages=0.5)
		>>>
		>>> # Scroll a specific element
		>>> action = ScrollAction(down=True, num_pages=2.0, frame_element_index=10)
	"""
	down: bool  # True to scroll down, False to scroll up
	num_pages: float  # Number of pages to scroll (0.5 = half page, 1.0 = one page, etc.)
	frame_element_index: int | None = None  # Optional element index to find scroll container for


class SendKeysAction(BaseModel):
	"""Action model for sending keyboard input.
	
	@public
	
	Sends keyboard keys or key combinations to the browser. Useful for
	shortcuts, special keys, or text input without a specific target element.
	
	Args:
		keys: The keyboard keys/key combinations to send.
	
	Example:
		>>> # Send Enter key
		>>> action = SendKeysAction(keys="Enter")
		>>>
		>>> # Send keyboard shortcut
		>>> action = SendKeysAction(keys="cmd+a")
		>>>
		>>> # Type text directly
		>>> action = SendKeysAction(keys="Hello World")
	"""
	keys: str


class UploadFileAction(BaseModel):
	"""Action model for uploading a file to a form element.
	
	@public
	
	Uploads a file to a file input element on the page.
	
	Args:
		index: The index of the file input element to upload to.
		path: The file system path to the file to upload.
	
	Example:
		>>> # Upload a file to the 2nd file input
		>>> action = UploadFileAction(
		...     index=2,
		...     path="/path/to/document.pdf"
		... )
	"""
	index: int
	path: str


class ExtractPageContentAction(BaseModel):
	"""Action model for extracting content from the current page.
	
	@public
	
	Extracts specific content from the current page based on a selector
	or extraction specification.
	
	Args:
		value: The content extraction specification or selector.
	
	Example:
		>>> # Extract specific content
		>>> action = ExtractPageContentAction(value=".article-content")
	"""
	value: str


class NoParamsAction(BaseModel):
	"""Action model for actions that don't require parameters.
	
	@public
	
	Accepts absolutely anything in the incoming data and discards it,
	so the final parsed model is empty. Used for actions like go_back()
	that don't need any input parameters.
	
	This model is used internally by the framework for parameter-less actions.
	
	Example:
		>>> # Used internally by actions like:
		>>> action = ActionModel(go_back=NoParamsAction())
	"""

	model_config = ConfigDict(extra='ignore')
	# No fields defined - all inputs are ignored automatically


class GetDropdownOptionsAction(BaseModel):
	"""Action model for retrieving available options from a dropdown element.
	
	@public
	
	Retrieves all available options from a dropdown/select element.
	
	Args:
		index: The index of the dropdown element to get options for (1-based indexing).
	
	Example:
		>>> # Get options from the 3rd dropdown
		>>> action = GetDropdownOptionsAction(index=3)
	"""
	index: int = Field(ge=1, description='index of the dropdown element to get the option values for')


class SelectDropdownOptionAction(BaseModel):
	"""Action model for selecting an option from a dropdown element.
	
	@public
	
	Selects a specific option from a dropdown/select element.
	
	Args:
		index: The index of the dropdown element (1-based indexing).
		text: The text or exact value of the option to select.
	
	Example:
		>>> # Select an option from the 2nd dropdown
		>>> action = SelectDropdownOptionAction(
		...     index=2,
		...     text="United States"
		... )
	"""
	index: int = Field(ge=1, description='index of the dropdown element to select an option for')
	text: str = Field(description='the text or exact value of the option to select')
