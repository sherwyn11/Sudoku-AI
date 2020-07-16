from preprocessing.preprocess import Preprocess
from recognition.recognition import DigitRecognition
from solver.solver import Solver
from helpers.constants import IMAGE_PATH


def preprocess_uploaded_image(image_file):
    preprocess = Preprocess(3, 1)
    preprocess.read_img(image_file)
    conv_img = preprocess.threshold_and_invert()
    dilated_image = preprocess.dialate_image(conv_img)
    outerbox, height, width = preprocess.flood_fill_image(dilated_image)
    erode = preprocess.erode_image(outerbox)
    line_img, lines = preprocess.draw_lines_on_image(erode)
    lin_image, tmpimg, leftedge, rightedge, topedge, bottomedge = preprocess.find_extreme_lines(lines, line_img)
    lin_image, detTopLeft, ptTopLeft, detTopRight, ptTopRight, detBottomRight, ptBottomRight, detBottomLeft, ptBottomLeft = preprocess.calculate_points(lin_image, leftedge, rightedge, topedge, bottomedge, height, width)
    preprocess.find_max_side_len(lin_image, detTopLeft, ptTopLeft, detTopRight, ptTopRight, detBottomRight, ptBottomRight, detBottomLeft, ptBottomLeft)
    cells = preprocess.create_image_grid()

    return cells


def get_digits(cells):
    recognize = DigitRecognition()
    recognize.load_keras_model()
    digits = recognize.get_digits(cells)

    return digits

def solve_sudoku(digits):
    solver = Solver()
    solver.load_solver_model('model.h5')
    solved = solver.solve(digits)

    return solved
