import cv2

from face_service.detector import FaceDetector
from face_service.embedding import FaceEmbedder


print("Starting YuNet...")
detector = FaceDetector()

print("Starting SFace...")
embedder = FaceEmbedder()

print("Starting camera...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open camera")


while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame")
        break

    faces = detector.detect(frame)

    if len(faces) == 0:

        cv2.putText(
            frame,
            "NO FACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    else:

        face = faces[0]

        x, y, w, h = face[:4].astype(int)

        try:

            embedding = embedder.extract(
                frame,
                face
            )

            print(
                "Embedding generated:",
                embedding.shape
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "EMBEDDING GENERATED",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        except Exception as e:

            print("SFace error:", repr(e))

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "SFACE ERROR",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    cv2.imshow(
        "SFace Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()