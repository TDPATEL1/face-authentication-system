import cv2

from face_service.detector import FaceDetector


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open camera")

detector = FaceDetector()

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break

    faces = detector.detect(frame)

    print("Faces detected:", len(faces))

    for face in faces:
        x, y, w, h = face[:4].astype(int)

        confidence = float(face[14])

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"YuNet {confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Draw the 5 landmarks
        for i in range(5):
            landmark_x = int(face[4 + i * 2])
            landmark_y = int(face[5 + i * 2])

            cv2.circle(
                frame,
                (landmark_x, landmark_y),
                3,
                (255, 0, 0),
                -1
            )

    cv2.imshow("YuNet Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()