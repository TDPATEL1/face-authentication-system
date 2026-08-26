import cv2
import numpy as np

from face_service.detector import FaceDetector
from face_service.embedding import FaceEmbedder


detector = FaceDetector()
embedder = FaceEmbedder()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open camera.")

print("Camera started.")
print("Press Q to quit.")


embedding_generated = False

while True:

    ret, frame = camera.read()

    if not ret:
        print("Failed to read camera frame.")
        break

    faces = detector.detect(frame)

    if len(faces) > 0:

        face = faces[0]

        x, y, w, h = face[:4].astype(int)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        if not embedding_generated:

            try:

                embedding = embedder.extract(
                    frame,
                    face
                )

                print("\n==============================")
                print("EMBEDDING GENERATED")
                print("Shape:", embedding.shape)
                print("Data type:", embedding.dtype)
                print("First 10 values:")
                print(embedding[0][:10])
                print("==============================\n")

                embedding_generated = True

            except Exception as e:

                print("Embedding error:")
                print(e)

        cv2.putText(
            frame,
            "FACE DETECTED",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        embedding_generated = False

        cv2.putText(
            frame,
            "NO FACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Face Embedding Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()