import cv2

from face_service.detector import FaceDetector
from face_service.liveness import LivenessDetector


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open camera")


detector = FaceDetector()
liveness = LivenessDetector()


while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break

    faces = detector.detect(frame)

    for face in faces:

        x, y, w, h = face

        live = liveness.check(face)

        if live:
            status = "LIVE"
        else:
            status = "MOVE HEAD"

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            status,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

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

    cv2.imshow(
        "Face Gate Camera",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()