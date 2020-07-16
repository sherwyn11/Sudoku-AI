import numpy as np
import os
import cv2

from helpers.constants import MODEL_PATH, DIGITS_PATH
from helpers.helpers import preprocess_image, check_if_white
from keras.models import load_model

class DigitRecognition:

    def __init__(self):
        self.digits = []
        self.model = None

    def load_keras_model(self):
        self.model = load_model(MODEL_PATH + 'model_digits99.h5')

    def get_digits(self, cells):
        # img_list = os.listdir(DIGITS_PATH)
        # img_list.sort()
        i = 0
        j = 0
        for img in cells:
            # image = cv2.imread(DIGITS_PATH + img, 0)
            image = preprocess_image(img, i, j)
            if j % 9 == 0 and j != 0:
                i += 1
                j = 0
            else:
                j += 1
            
            does_image_contain_white = check_if_white(image)
            if does_image_contain_white:
                image = image.reshape((1, 28, 28, 1))
                number = self.model.predict(image)
                number = np.argmax(number)
                self.digits.append(number)
            else:
                self.digits.append(0)

        return self.digits