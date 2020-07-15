from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return 'Test Works!'


@app.route('/about')
def about():
    return render_template('pages/placeholder.about.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)