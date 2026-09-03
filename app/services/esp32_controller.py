import logging
from typing import Optional

import requests


logger = logging.getLogger(__name__)


class ESP32DoorController:
    """
    Controller used by FastAPI to communicate with an ESP32.

    The ESP32 is expected to expose:

        POST /unlock
        POST /lock
        GET  /status
    """

    def __init__(
        self,
        esp32_ip: str,
        timeout: float = 3.0,
    ):
        self.esp32_ip = esp32_ip.rstrip("/")
        self.timeout = timeout

    def unlock(self) -> bool:
        """
        Send an unlock command to the ESP32.
        """

        url = f"{self.esp32_ip}/unlock"

        try:
            response = requests.post(
                url,
                timeout=self.timeout,
            )

            response.raise_for_status()

            data = response.json()

            success = data.get("success", False)

            if success:
                logger.info("ESP32 door unlock command successful")
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
        Send a lock command to the ESP32.
        """

        url = f"{self.esp32_ip}/lock"

        try:
            response = requests.post(
                url,
                timeout=self.timeout,
            )

            response.raise_for_status()

            data = response.json()

            success = data.get("success", False)

            if success:
                logger.info("ESP32 door lock command successful")
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
        """

        url = f"{self.esp32_ip}/status"

        try:
            response = requests.get(
                url,
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