import numpy as np
import cv2

from helpers.constants import MODEL_PATH, DIGITS_PATH, DIGIT_RECOGNIZER_MODEL_NAME
from helpers.helpers import preprocess_image, check_if_white
from keras.models import load_model


class DigitRecognition:

    def __init__(self):
        '''
        Init function
        '''

        self.digits = []
        self.model = None


    def load_keras_model(self):
        '''
        Load the trained Keras model for Digit Recognition
        '''

        self.model = load_model(MODEL_PATH + DIGIT_RECOGNIZER_MODEL_NAME)


    def get_digits(self, cells):
        '''
        Extract the digits from each cell of the Sudoku board
        '''

        for img in cells:
            image = preprocess_image(img)
            does_image_contain_white = check_if_white(image)

            if does_image_contain_white:
                image = image.reshape((1, 28, 28, 1))
                number = self.model.predict(image)
                number = np.argmax(number)
                self.digits.append(number)
            else:
                self.digits.append(0)

        return self.digits