import random
import time
from typing import Optional


class LivenessDetector:
    """
    Directional challenge-based liveness detector.

    The detector observes the movement of a detected face
    across multiple camera frames.

    A random challenge is generated when the detector starts:

        TURN_LEFT
        TURN_RIGHT

    The user must move their face in the requested direction.

    NOTE:
        This is still a basic anti-spoofing mechanism.
        It is stronger than simple movement detection, but it is
        NOT production-grade liveness detection against advanced
        replay, deepfake, or 3D mask attacks.
    """

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    MOVEMENT_THRESHOLD = 5.0

    # Number of meaningful movements required.
    REQUIRED_MOVEMENTS = 2

    # How long successful liveness remains valid.
    LIVENESS_VALID_SECONDS = 10.0

    # Maximum time allowed for one liveness session.
    SESSION_TIMEOUT_SECONDS = 60.0

    # Minimum directional movement required.
    DIRECTION_THRESHOLD = 4.0

    # ---------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------

    def __init__(
        self,
        challenge: Optional[str] = None,
    ):
        self.previous_position: Optional[
            tuple[float, float]
        ] = None

        self.movement_count = 0

        self.live_until = 0.0

        self.started_at = time.monotonic()

        self.completed = False

        # Generate a random challenge if one was not supplied.
        self.challenge = (
            challenge
            if challenge
            else random.choice(
                [
                    "TURN_LEFT",
                    "TURN_RIGHT",
                ]
            )
        )

    # ---------------------------------------------------------
    # Reset detector
    # ---------------------------------------------------------

    def reset(self) -> None:
        """
        Reset the detector and generate a new challenge.
        """

        self.previous_position = None

        self.movement_count = 0

        self.live_until = 0.0

        self.started_at = time.monotonic()

        self.completed = False

        self.challenge = random.choice(
            [
                "TURN_LEFT",
                "TURN_RIGHT",
            ]
        )

    # ---------------------------------------------------------
    # Session expiration
    # ---------------------------------------------------------

    def is_expired(self) -> bool:
        """
        Return True if the liveness session has timed out.
        """

        elapsed = (
            time.monotonic()
            - self.started_at
        )

        return (
            elapsed
            > self.SESSION_TIMEOUT_SECONDS
        )

    # ---------------------------------------------------------
    # Liveness status
    # ---------------------------------------------------------

    def is_live(self) -> bool:
        """
        Return True while successful liveness remains valid.
        """

        if not self.completed:
            return False

        return (
            time.monotonic()
            < self.live_until
        )

    # ---------------------------------------------------------
    # Challenge text
    # ---------------------------------------------------------

    def challenge_message(self) -> str:
        """
        Return a human-readable instruction.
        """

        if self.challenge == "TURN_LEFT":
            return (
                "Please slowly move your face to the left."
            )

        if self.challenge == "TURN_RIGHT":
            return (
                "Please slowly move your face to the right."
            )

        return "Please move your face."

    # ---------------------------------------------------------
    # Process one face frame
    # ---------------------------------------------------------

    def check(self, face) -> bool:
        """
        Process one detected face.

        Args:
            face:
                YuNet face detection result:
                [x, y, width, height, ...]

        Returns:
            True  -> liveness passed
            False -> liveness not yet passed
        """

        # -----------------------------------------------------
        # Validate session
        # -----------------------------------------------------

        if self.is_expired():
            return False

        # -----------------------------------------------------
        # Already live
        # -----------------------------------------------------

        if self.is_live():
            return True

        # -----------------------------------------------------
        # Validate face
        # -----------------------------------------------------

        if face is None:
            return False

        try:
            x = float(face[0])
            y = float(face[1])
            w = float(face[2])
            h = float(face[3])

        except (
            TypeError,
            ValueError,
            IndexError,
        ):
            return False

        if w <= 0 or h <= 0:
            return False

        # -----------------------------------------------------
        # Calculate face center
        # -----------------------------------------------------

        current_x = (
            x
            + (w / 2.0)
        )

        current_y = (
            y
            + (h / 2.0)
        )

        # -----------------------------------------------------
        # First frame
        # -----------------------------------------------------

        if self.previous_position is None:

            self.previous_position = (
                current_x,
                current_y,
            )

            return False

        # -----------------------------------------------------
        # Previous position
        # -----------------------------------------------------

        previous_x, previous_y = (
            self.previous_position
        )

        # Signed movement is important here.

        delta_x = (
            current_x
            - previous_x
        )

        delta_y = (
            current_y
            - previous_y
        )

        movement_x = abs(delta_x)

        movement_y = abs(delta_y)

        total_movement = (
            movement_x
            + movement_y
        )

        # Update position.

        self.previous_position = (
            current_x,
            current_y,
        )

        # -----------------------------------------------------
        # Check directional movement
        # -----------------------------------------------------

        correct_direction = False

        if (
            self.challenge
            == "TURN_LEFT"
        ):

            # Face center must move left.

            if (
                delta_x
                <= -self.DIRECTION_THRESHOLD
            ):
                correct_direction = True

        elif (
            self.challenge
            == "TURN_RIGHT"
        ):

            # Face center must move right.

            if (
                delta_x
                >= self.DIRECTION_THRESHOLD
            ):
                correct_direction = True

        # -----------------------------------------------------
        # Count valid movement
        # -----------------------------------------------------

        if (
            total_movement
            >= self.MOVEMENT_THRESHOLD
            and correct_direction
        ):

            self.movement_count += 1

        else:

            self.movement_count = max(
                0,
                self.movement_count - 1,
            )

        # -----------------------------------------------------
        # Liveness passed
        # -----------------------------------------------------

        if (
            self.movement_count
            >= self.REQUIRED_MOVEMENTS
        ):

            self.completed = True

            self.live_until = (
                time.monotonic()
                + self.LIVENESS_VALID_SECONDS
            )

            self.movement_count = 0

            return True

        return False

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------

    def status(self) -> dict:
        """
        Return current liveness state.
        """

        return {
            "live": self.is_live(),

            "completed": self.completed,

            "movement_count":
                self.movement_count,

            "challenge":
                self.challenge,

            "challenge_message":
                self.challenge_message(),

            "expired":
                self.is_expired(),

            "remaining_seconds": (
                max(
                    0.0,
                    self.live_until
                    - time.monotonic(),
                )
                if self.completed
                else 0.0
            ),
        }