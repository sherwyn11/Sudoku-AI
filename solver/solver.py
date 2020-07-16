import numpy as np

from helpers.constants import MODEL_PATH
from helpers.helpers import normalize_data, denormalize_data
from keras.models import load_model

class Solver:

    def __init__(self):
        self.output = []
        self.model = None
    

    def load_solver_model(self, model_name):
        self.model = load_model(MODEL_PATH + model_name)


    def solve_sudoku(self, digits):

        digits = np.array(digits).reshape((9, 9, 1))
        digits = normalize_data(digits)

        input_copy = np.copy(digits)

        while(True):
        
            self.output = self.model.predict(input_copy.reshape((1,9,9,1)))  
            self.output = self.output.squeeze()

            predicted_idxs = np.argmax(self.output, axis=1).reshape((9,9))+1 
            prob_values = np.around(np.max(self.output, axis=1).reshape((9,9)), 2) 
            
            input_copy = denormalize_data(input_copy).reshape((9,9))
            mask = (input_copy==0)
        
            if(mask.sum()==0):
                break
                
            prob_new = prob_values * mask
        
            ind = np.argmax(prob_new)
            x, y = (ind//9), (ind%9)

            val = predicted_idxs[x][y]
            input_copy[x][y] = val
            input_copy = normalize_data(input_copy)
        
        return predicted_idxs.flatten()