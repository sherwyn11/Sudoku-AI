from flask import Flask, render_template, request
from utils.utils import *


app = Flask(__name__)

@app.route('/')
def home():
    '''
    Testing Flask Server
    '''
    return 'Test Works!'


@app.route('/solve', methods=['GET', 'POST'])
def solve():
    '''
    Solve Route
    '''

    if request.method == 'POST':
        sudoku_image = request.files['image'].read()
        cells = preprocess_uploaded_image(sudoku_image)
        digits = get_digits_from_cells(cells)
        sudoku_solved = solve_sudoku(digits)

        return render_template('index.html', data=sudoku_solved, posted=1)
    else:
        return render_template('index.html', posted=0)


if __name__ == '__main__':
    '''
    Run Flask Server
    '''

    app.run(host='0.0.0.0', port=5000, debug=True)