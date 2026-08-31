import logging
import threading

logger = logging.getLogger(__name__)


class DoorController:
    """
    Abstract interface for controlling the physical door.

    The first implementation is simulation-only.
    Later, an ESP32 controller can implement the same interface.
    """

    def unlock(self) -> bool:
        raise NotImplementedError

    def lock(self) -> bool:
        raise NotImplementedError

    def is_unlocked(self) -> bool:
        raise NotImplementedError


class SimulationDoorController(DoorController):
    """
    Software-only door controller.

    Used while developing/testing without physical hardware.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._unlocked = False

    def unlock(self) -> bool:
        with self._lock:
            self._unlocked = True

        logger.info("SIMULATION: DOOR UNLOCKED")
        return True

    def lock(self) -> bool:
        with self._lock:
            self._unlocked = False

        logger.info("SIMULATION: DOOR LOCKED")
        return True

    def is_unlocked(self) -> bool:
        with self._lock:
            return self._unlocked