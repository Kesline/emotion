import numpy as np
import cv2
import dlib
from imutils import face_utils
import torch
from config import SHAPE_PREDICTOR_PATH, YOLOV5_MODEL_PATH


class EmotionDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=YOLOV5_MODEL_PATH
        )

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return faces

    def predict_emotion(self, face, frame):
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cropped_face = frame[y : y + h, x : x + w]

        input_img = cv2.resize(cropped_face, (224, 224))
        input_img = np.expand_dims(input_img, axis=0)
        input_img = torch.tensor(input_img, dtype=torch.float32)

        results = self.model(input_img)
        emotion = results.names[
            results.pred[0].argmax()
        ]  # Assuming model returns a named prediction
        return emotion

    def draw_results(self, frame, face, emotion):
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2
        )

    def detect_and_recognize(self, frame):
        faces = self.detect_faces(frame)
        for face in faces:
            emotion = self.predict_emotion(face, frame)
            self.draw_results(frame, face, emotion)
        return frame


def main():
    cap = cv2.VideoCapture(0)  # Open the first webcam

    detector = EmotionDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.detect_and_recognize(frame)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
