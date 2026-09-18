import cv2

from face_service.detector import FaceDetector
from face_service.liveness import LivenessDetector


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera")

    detector = FaceDetector()
    liveness = LivenessDetector()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read frame")
                break

            faces = detector.detect(frame)

            for face in faces:
                # YuNet returns more than just x, y, w, h.
                # The first four values are the bounding box.
                x, y, w, h = face[:4]

                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                # Liveness detector works with the bounding box.
                bbox = [x, y, w, h]

                live = liveness.check(bbox)

                if live:
                    status = "LIVE"
                else:
                    status = "MOVE HEAD"

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    status,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            if len(faces) == 0:
                cv2.putText(
                    frame,
                    "NO FACE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow(
                "Face Gate Camera",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
