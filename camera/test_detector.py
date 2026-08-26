import cv2

from face_service.detector import FaceDetector


detector = FaceDetector()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open camera.")

print("Camera started.")
print("Press Q to quit.")

while True:

    ret, frame = camera.read()

    if not ret:
        print("Failed to read camera frame.")
        break

    faces = detector.detect(frame)

    print("Faces detected:", len(faces))

    for face in faces:

        x, y, w, h = face[:4].astype(int)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        confidence = face[-1]

        cv2.putText(
            frame,
            f"Face {confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "YuNet Face Detector",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()