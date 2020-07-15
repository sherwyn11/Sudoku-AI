import numpy as np
import os
import cv2

from helpers.constants import MODEL_PATH, DIGITS_PATH
from helpers.helpers import preprocess_image
from keras.models import load_model

class DigitRecognition:

    def __init__(self):
        self.digits = []
        self.model = None

    def load_keras_model(self):
        self.model = load_model(MODEL_PATH + 'model_digits.h5')

    def get_digits(self):
        img_list = os.listdir(DIGITS_PATH)
        img_list.sort()
        for img in img_list:
            image = cv2.imread(DIGITS_PATH + img, 0)
            image = preprocess_image(image)
            image = image.reshape((1, 28, 28, 1))
            number = self.model.predict(image)
            number = np.argmax(number)
            self.digits.append(number)

        print(self.digits)