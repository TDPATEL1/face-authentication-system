import cv2
import numpy as np

from face_service.detector import FaceDetector
from face_service.embedding import FaceEmbedder
from app.services.face_matching import FaceMatchingService


detector = FaceDetector()
embedder = FaceEmbedder()
matcher = FaceMatchingService()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open camera")

print("Camera started.")
print("Press SPACE to capture.")
print("Press Q to quit.")

while True:

    ret, frame = camera.read()

    if not ret:
        print("Failed to read camera frame")
        break

    cv2.imshow(
        "Face Matching Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord(" "):

        print("Capturing face...")

        try:

            embedding = matcher.extract_embedding(
                frame
            )

            print(
                "Embedding shape:",
                embedding.shape
            )

            print(
                "Embedding dtype:",
                embedding.dtype
            )

            print(
                "Embedding sample:",
                embedding.flatten()[:5]
            )

            print(
                "Face embedding generated successfully."
            )

        except ValueError as exc:

            print(
                "Face error:",
                exc
            )

camera.release()
cv2.destroyAllWindows()