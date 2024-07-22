import cv2
import dlib
import torch
import os


# Define paths
SHAPE_PREDICTOR_FILENAME = 'shape_predictor_68_face_landmarks.dat'
SHAPE_PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), SHAPE_PREDICTOR_FILENAME)
YOLOV5_MODEL_PATH = 'path_to_yolov5_model'


class EmotionDetector:
   def __init__(self, dataset_path):
       self.detector = dlib.get_frontal_face_detector()
       self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
       self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)


       # Initialize dataset loading from local directory
       self.emotion_labels = ['Neutral', 'Happy', 'Sad', 'Angry']  # Updated emotion labels
       self.emotion_images = self.load_dataset(dataset_path)


       # Load Haar Cascade for face detection
       self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


   def load_dataset(self, dataset_path):
       emotion_images = {label: [] for label in self.emotion_labels}


       for label in self.emotion_labels:
           label_path = os.path.join(dataset_path, label)
           if os.path.exists(label_path):
               for img_file in os.listdir(label_path):
                   img_path = os.path.join(label_path, img_file)
                   emotion_images[label].append(cv2.imread(img_path))  # Assuming using OpenCV to read images
           else:
               print(f"Warning: Directory not found for label '{label}': {label_path}")


       return emotion_images


   def detect_faces(self, frame):
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       faces = self.detector(gray)
       return faces


   def detect_and_recognize(self, frame):
       # Convert frame to RGB (assuming OpenCV BGR format)
       frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


       # Perform object detection with YOLOv5
       results = self.model(frame_rgb)


       # Extract detections from the results tensor
       detections = results.pandas().xyxy[0]  # Assuming batch size 1


       # Process each detection
       for _, detection in detections.iterrows():
           x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
           conf = float(detection['confidence'])
           cls = int(detection['class'])


           # Crop face from the original frame (not resized)
           face = frame[y1:y2, x1:x2]


           # Perform emotion recognition on the face
           emotion = self.predict_emotion(face)


           # Draw bounding box and label on the frame
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
           cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


       return frame


   def predict_emotion(self, face):
       # Convert face to grayscale
       gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)


       # Detect faces in the grayscale image
       faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)


       # Placeholder for emotion prediction
       emotion = "Neutral"


       # Iterate over detected faces
       for (x, y, w, h) in faces:
           # Extract ROI of face
           roi_gray = gray[y:y+h, x:x+w]


           # Perform more advanced emotion recognition (e.g., using deep learning models)
           # Example: Replace with your actual emotion recognition model or logic
           # Here's a placeholder using a simple intensity-based heuristic
           roi_avg_intensity = np.mean(roi_gray)


           # Adjust thresholds based on your trained model or heuristic
           if roi_avg_intensity < 100:
               emotion = "Neutral"
           elif roi_avg_intensity < 120:
               emotion = "Happy"
           elif roi_avg_intensity < 140:
               emotion = "Sad"
           else:
               emotion = "Angry"


       return emotion


def main():
   # Specify the path to your dataset
   dataset_path = '/home/developer/PycharmProjects/pythonProject/train'


   # Initialize the EmotionDetector with the dataset path
   detector = EmotionDetector(dataset_path)


   # Initialize video capture (assuming webcam)
   cap = cv2.VideoCapture(0)


   while True:
       ret, frame = cap.read()
       if not ret:
           break


       # Detect and recognize emotions
       frame = detector.detect_and_recognize(frame)


       # Display the resulting frame
       cv2.imshow('Emotion Detection', frame)


       # Exit on 'q' key press
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break


   # Release the capture and close all windows
   cap.release()
   cv2.destroyAllWindows()


if __name__ == '__main__':
   main()
