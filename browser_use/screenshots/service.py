"""Screenshot storage service for browser-use agents.
"""

import base64
from pathlib import Path

import anyio


class ScreenshotService:
	"""Screenshot storage service for persisting browser screenshots.
	
	@public
	
	Provides methods to save and retrieve screenshots from disk. Screenshots are
	stored as PNG files in a dedicated directory and can be accessed by their
	file path or loaded back as base64-encoded strings.
	
	This service is useful for:
	- Persisting screenshots across agent steps
	- Creating visual logs of browser interactions
	- Saving screenshots for debugging or analysis
	- Managing screenshot storage lifecycle
	
	Attributes:
		agent_directory: Root directory for agent data
		screenshots_dir: Subdirectory where screenshots are stored
	
	Example:
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
	"""

	def __init__(self, agent_directory: str | Path):
		"""Initialize screenshot service with storage directory.
		
		@public
		
		Creates a screenshot service that saves screenshots to the specified
		directory. Automatically creates a 'screenshots' subdirectory if it
		doesn't exist.
		
		Args:
			agent_directory: Root directory for agent data. Can be a string path
				or Path object. Screenshots will be stored in a 'screenshots'
				subdirectory within this location.
		
		Example:
			>>> # Create service with string path
			>>> service = ScreenshotService("./my_agent")
			>>> 
			>>> # Or with Path object
			>>> from pathlib import Path
			>>> service = ScreenshotService(Path.home() / "agent_data")
		"""
		self.agent_directory = Path(agent_directory) if isinstance(agent_directory, str) else agent_directory

		# Create screenshots subdirectory
		self.screenshots_dir = self.agent_directory / 'screenshots'
		self.screenshots_dir.mkdir(parents=True, exist_ok=True)

	async def store_screenshot(self, screenshot_b64: str, step_number: int) -> str:
		"""Store a base64-encoded screenshot to disk.
		
		@public
		
		Saves a screenshot as a PNG file with a filename based on the step number.
		The screenshot is decoded from base64 and written to disk in the screenshots
		directory.
		
		Args:
			screenshot_b64: Base64-encoded PNG screenshot data, as returned by
				browser_session.get_browser_state_summary(include_screenshot=True)
				or ScreenshotEvent.
			step_number: The step number for naming the file (e.g., step_1.png).
				Used to organize screenshots chronologically.
		
		Returns:
			Full file path to the saved screenshot as a string.
		
		Example:
			>>> # Get screenshot from browser state
			>>> state = await browser_session.get_browser_state_summary(include_screenshot=True)
			>>> 
			>>> # Save it to disk
			>>> path = await screenshot_service.store_screenshot(
			...     screenshot_b64=state.screenshot,
			...     step_number=1
			... )
			>>> print(f"Saved: {path}")  # e.g., "./agent_output/screenshots/step_1.png"
		"""
		screenshot_filename = f'step_{step_number}.png'
		screenshot_path = self.screenshots_dir / screenshot_filename

		# Decode base64 and save to disk
		screenshot_data = base64.b64decode(screenshot_b64)

		async with await anyio.open_file(screenshot_path, 'wb') as f:
			await f.write(screenshot_data)

		return str(screenshot_path)

	async def get_screenshot(self, screenshot_path: str) -> str | None:
		"""Load a screenshot from disk and return as base64.
		
		@public
		
		Reads a previously saved screenshot from disk and returns it as a
		base64-encoded string. This can be used to reload screenshots for
		further processing or to send to AI vision models.
		
		Args:
			screenshot_path: Full file path to the screenshot file, as returned
				by store_screenshot().
		
		Returns:
			Base64-encoded PNG data as a string, or None if the file doesn't exist
			or the path is empty.
		
		Example:
			>>> # Load a previously saved screenshot
			>>> screenshot_b64 = await screenshot_service.get_screenshot(
			...     "./agent_output/screenshots/step_1.png"
			... )
			>>> 
			>>> if screenshot_b64:
			...     # Use with AI vision model
			...     vision_input = f"data:image/png;base64,{screenshot_b64}"
		"""
		if not screenshot_path:
			return None

		path = Path(screenshot_path)
		if not path.exists():
			return None

		# Load from disk and encode to base64
		async with await anyio.open_file(path, 'rb') as f:
			screenshot_data = await f.read()

		return base64.b64encode(screenshot_data).decode('utf-8')
