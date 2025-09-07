"""Telemetry service for usage analytics and monitoring."""
import logging
import os

from dotenv import load_dotenv
from posthog import Posthog
from uuid_extensions import uuid7str

from browser_use.telemetry.views import BaseTelemetryEvent
from browser_use.utils import singleton

load_dotenv()

from browser_use.config import CONFIG

logger = logging.getLogger(__name__)


POSTHOG_EVENT_SETTINGS = {
	'process_person_profile': True,
}


@singleton
class ProductTelemetry:
	"""Service for capturing anonymized telemetry data.
	
	@public
	
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
	
	Note:
		Telemetry helps us understand usage patterns and improve the library.
		It's completely anonymous and can be disabled at any time.
	"""

	USER_ID_PATH = str(CONFIG.BROWSER_USE_CONFIG_DIR / 'device_id')
	PROJECT_API_KEY = 'phc_F8JMNjW1i2KbGUTaW1unnDdLSPCoyc52SGRU0JecaUh'
	HOST = 'https://eu.i.posthog.com'
	UNKNOWN_USER_ID = 'UNKNOWN'

	_curr_user_id = None

	def __init__(self) -> None:
		telemetry_disabled = not CONFIG.ANONYMIZED_TELEMETRY
		self.debug_logging = CONFIG.BROWSER_USE_LOGGING_LEVEL == 'debug'

		if telemetry_disabled:
			self._posthog_client = None
		else:
			logger.info('Using anonymized telemetry, see https://docs.browser-use.com/development/telemetry.')
			self._posthog_client = Posthog(
				project_api_key=self.PROJECT_API_KEY,
				host=self.HOST,
				disable_geoip=False,
				enable_exception_autocapture=True,
			)

			# Silence posthog's logging
			if not self.debug_logging:
				posthog_logger = logging.getLogger('posthog')
				posthog_logger.disabled = True

		if self._posthog_client is None:
			logger.debug('Telemetry disabled')

	def capture(self, event: BaseTelemetryEvent) -> None:
		"""Capture a telemetry event if client is available."""
		if self._posthog_client is None:
			return

		self._direct_capture(event)

	def _direct_capture(self, event: BaseTelemetryEvent) -> None:
		"""Should not be thread blocking because posthog magically handles it
		"""
		if self._posthog_client is None:
			return

		try:
			self._posthog_client.capture(
				distinct_id=self.user_id,
				event=event.name,
				properties={**event.properties, **POSTHOG_EVENT_SETTINGS},
			)
		except Exception as e:
			logger.error(f'Failed to send telemetry event {event.name}: {e}')

	def flush(self) -> None:
		"""Flush any pending telemetry events."""
		if self._posthog_client:
			try:
				self._posthog_client.flush()
				logger.debug('PostHog client telemetry queue flushed.')
			except Exception as e:
				logger.error(f'Failed to flush PostHog client: {e}')
		else:
			logger.debug('PostHog client not available, skipping flush.')

	@property
	def user_id(self) -> str:
		if self._curr_user_id:
			return self._curr_user_id

		# File access may fail due to permissions or other reasons. We don't want to
		# crash so we catch all exceptions.
		try:
			if not os.path.exists(self.USER_ID_PATH):
				os.makedirs(os.path.dirname(self.USER_ID_PATH), exist_ok=True)
				with open(self.USER_ID_PATH, 'w') as f:
					new_user_id = uuid7str()
					f.write(new_user_id)
				self._curr_user_id = new_user_id
			else:
				with open(self.USER_ID_PATH) as f:
					self._curr_user_id = f.read()
		except Exception:
			self._curr_user_id = 'UNKNOWN_USER_ID'
		return self._curr_user_id
