from ..detectionModelAdaptor.detectionModelAdaptor import DetectionModelAdaptor
from ...servicesHandler.servicesHandler import ServicesHandler
import cv2
from fer import FER

class EmotionModel(DetectionModelAdaptor):
    
    def __init__(self):
        self.__services_handler = ServicesHandler()

    def predict (self, frame):
        emotion_result, emotion_score = self.__services_handler.handle_emotion_predict(frame)
        # print(f"emotion model emotion result: {emotion_result}")
        return emotion_result, emotion_score
