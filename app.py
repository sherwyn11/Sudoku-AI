from flask import Flask, render_template, request
from utils.utils import *

app = Flask(__name__)

@app.route('/')
def home():
    return 'Test Works!'


@app.route('/solve', methods=['POST'])
def solved():
    sudoku_image = request.files['image'].read()
    cells = preprocess_uploaded_image(sudoku_image)
    digits = get_digits(cells)
    sudoku_solved = solve_sudoku(digits)
    print(sudoku_solved)

    return "Done"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)