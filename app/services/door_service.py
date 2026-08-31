import logging
import threading

from app.services.door_controller import (
    DoorController,
    SimulationDoorController,
)

logger = logging.getLogger(__name__)


class DoorService:
    """
    High-level door business logic.

    The controller handles the actual hardware.
    DoorService handles timing and application logic.
    """

    def __init__(
        self,
        controller: DoorController | None = None,
        unlock_duration: float = 5.0,
    ):
        self.controller = controller or SimulationDoorController()
        self.unlock_duration = unlock_duration

        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def unlock(self) -> bool:
        """
        Unlock the door temporarily.
        """

        with self._lock:

            # Cancel an existing timer if the door
            # is already unlocked and someone authenticates again.
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

            success = self.controller.unlock()

            if not success:
                logger.error("Failed to unlock door")
                return False

            logger.info(
                "Door unlocked for %.1f seconds",
                self.unlock_duration,
            )

            self._timer = threading.Timer(
                self.unlock_duration,
                self.lock,
            )

            self._timer.daemon = True
            self._timer.start()

            return True

    def lock(self) -> bool:
        """
        Lock the door.
        """

        with self._lock:

            success = self.controller.lock()

            if not success:
                logger.error("Failed to lock door")
                return False

            self._timer = None

            logger.info("Door locked")

            return True

    def is_unlocked(self) -> bool:
        """
        Return current door state.
        """

        return self.controller.is_unlocked()