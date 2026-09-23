import logging
from typing import Optional

import requests

from app.core.config import settings
from app.services.door_controller import DoorController


logger = logging.getLogger(__name__)


class ESP32DoorController(DoorController):
    """
    Controller used by FastAPI to communicate with an ESP32.

    The ESP32 requires a shared secret in the
    X-Door-Authorization header.

    Expected endpoints:

        POST /unlock
        POST /lock
        GET  /status
    """

    def __init__(
        self,
        esp32_ip: str,
        timeout: float = 3.0,
        shared_secret: str | None = None,
    ):
        self.esp32_ip = esp32_ip.rstrip("/")
        self.timeout = timeout

        self.shared_secret = (
            shared_secret
            if shared_secret is not None
            else settings.ESP32_SHARED_SECRET
        )

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-Door-Authorization": self.shared_secret,
        }

    def unlock(self) -> bool:
        """
        Send an authenticated unlock command to the ESP32.
        """

        url = f"{self.esp32_ip}/unlock"

        try:
            response = requests.post(
                url,
                headers=self.headers,
                timeout=self.timeout,
            )

            response.raise_for_status()

            data = response.json()

            success = data.get("success", False)

            if success:
                logger.info(
                    "ESP32 door unlock command successful"
                )
                return True

            logger.warning(
                "ESP32 rejected unlock command: %s",
                data,
            )

            return False

        except requests.RequestException as exc:
            logger.error(
                "Could not communicate with ESP32: %s",
                exc,
            )
            return False

        except ValueError as exc:
            logger.error(
                "Invalid response received from ESP32: %s",
                exc,
            )
            return False

    def lock(self) -> bool:
        """
        Send an authenticated lock command to the ESP32.
        """

        url = f"{self.esp32_ip}/lock"

        try:
            response = requests.post(
                url,
                headers=self.headers,
                timeout=self.timeout,
            )

            response.raise_for_status()

            data = response.json()

            success = data.get("success", False)

            if success:
                logger.info(
                    "ESP32 door lock command successful"
                )
                return True

            logger.warning(
                "ESP32 rejected lock command: %s",
                data,
            )

            return False

        except requests.RequestException as exc:
            logger.error(
                "Could not communicate with ESP32: %s",
                exc,
            )
            return False

        except ValueError as exc:
            logger.error(
                "Invalid response received from ESP32: %s",
                exc,
            )
            return False

    def is_unlocked(self) -> bool:
        """
        Ask the ESP32 for the current door state.

        Status requests are also authenticated.
        """

        url = f"{self.esp32_ip}/status"

        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
            )

            response.raise_for_status()

            data = response.json()

            return bool(
                data.get("unlocked", False)
            )

        except requests.RequestException as exc:
            logger.error(
                "Could not get door status from ESP32: %s",
                exc,
            )
            return False

        except ValueError as exc:
            logger.error(
                "Invalid status response from ESP32: %s",
                exc,
            )
            return False