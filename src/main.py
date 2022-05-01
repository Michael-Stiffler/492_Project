from flask import Flask, render_template, redirect, url_for, request, make_response, jsonify
from flask import make_response
from waitress import serve

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('/index.html')


@app.route('/datapull', methods=['GET', 'POST'])
def datepull():
    if request.method == 'POST':
        datafromjs = request.form['data']
        print(datafromjs)

    resp = jsonify("This is the answer")
    return resp


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8000)
