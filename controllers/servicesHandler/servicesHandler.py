from fer import FER
import cv2

class ServicesHandler:

    
    def handle_emotion_predict(self, frame):
        emotion_detector = FER()
        emotions = emotion_detector.detect_emotions(frame)
        if emotions:
            emotion = emotions[0]['emotions']
            max_emotion = max(emotion, key=emotion.get)
            max_score = emotion[max_emotion]
            return max_emotion, max_score
        else:
            return None, None

    def handle_face_predict_request(self, frame):
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # print(f"services handler face result: {faces}")
            return faces
        