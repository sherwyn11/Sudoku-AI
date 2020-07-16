import numpy as np

from helpers.constants import MODEL_PATH
from keras.models import load_model

class Solver:

    def __init__(self):
        self.output = []
        self.model = None
    

    def load_solver_model(self, model_name):
        self.model = load_model(MODEL_PATH + model_name)


    def solve(self, digits):
        inp = np.array(digits).reshape((1, 9, 9, 1))
        inp = inp / 9
        inp -= .5    
        opt = self.model.predict(inp)
        
        for number in opt[0]:
            self.output.append(np.argmax(number) + 1)

        print(digits)
        return self.output