import time


class LivenessDetector:
    def __init__(self):
        self.previous_position = None
        self.movement_count = 0

        # Keep LIVE displayed for this many seconds
        self.live_until = 0

    def check(self, face):
        x, y, w, h = face

        current_x = x + w // 2
        current_y = y + h // 2

        # First frame
        if self.previous_position is None:
            self.previous_position = (current_x, current_y)
            return False

        previous_x, previous_y = self.previous_position

        movement_x = abs(current_x - previous_x)
        movement_y = abs(current_y - previous_y)

        self.previous_position = (current_x, current_y)

        # Detect movement
        if movement_x > 5 or movement_y > 5:
            self.movement_count += 1
        else:
            self.movement_count = max(
                0,
                self.movement_count - 1
            )

        # Movement detected enough times
        if self.movement_count >= 2:
            self.live_until = time.time() + 3
            self.movement_count = 0

        # Keep LIVE for 3 seconds
        if time.time() < self.live_until:
            return True

        return False