import sys
from pathlib import Path

# ============================================================
# Add project root to Python import path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import cv2

from face_service.detector import FaceDetector
from app.services.anti_spoofing import anti_spoofing_service


def main():
    print("=" * 60)
    print("MiniFASNetV2 Anti-Spoofing Camera Test")
    print("=" * 60)
    print("Starting camera...")
    print("Press Q to quit.")
    print()

    detector = FaceDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open laptop camera.")
        return

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        try:
            faces = detector.detect(frame)

        except Exception as exc:
            print(f"Face detection error: {exc}")
            break

        display_frame = frame.copy()

        # ========================================================
        # No face
        # ========================================================

        if len(faces) == 0:

            cv2.putText(
                display_frame,
                "NO FACE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        # ========================================================
        # Multiple faces
        # ========================================================

        elif len(faces) > 1:

            cv2.putText(
                display_frame,
                "MULTIPLE FACES",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        # ========================================================
        # Exactly one face
        # ========================================================

        else:

            face = faces[0]

            x = int(face[0])
            y = int(face[1])
            w = int(face[2])
            h = int(face[3])

            try:

                result = anti_spoofing_service.predict(
                    frame,
                    face,
                )

                is_real = result["is_real"]
                confidence = result["confidence"]
                class_id = result["class_id"]
                probabilities = result["probabilities"]

                # ------------------------------------------------
                # Face rectangle
                # ------------------------------------------------

                cv2.rectangle(
                    display_frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    2,
                )

                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------

                if is_real:
                    status = "REAL"
                else:
                    status = "SPOOF"

                cv2.putText(
                    display_frame,
                    f"Prediction: {status}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0) if is_real else (0, 0, 255),
                    2,
                )

                cv2.putText(
                    display_frame,
                    f"Confidence: {confidence:.4f}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    display_frame,
                    f"Class ID: {class_id}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # ------------------------------------------------
                # Three class probabilities
                # ------------------------------------------------

                for index, probability in enumerate(probabilities):

                    text = (
                        f"Class {index}: "
                        f"{probability:.4f}"
                    )

                    cv2.putText(
                        display_frame,
                        text,
                        (20, 150 + index * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

            except Exception as exc:

                cv2.putText(
                    display_frame,
                    "ANTI-SPOOF ERROR",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

                print(
                    f"Anti-spoofing prediction error: {exc}"
                )

        cv2.imshow(
            "MiniFASNetV2 Anti-Spoofing Test",
            display_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print()
    print("Camera test stopped.")


if __name__ == "__main__":
    main()
    