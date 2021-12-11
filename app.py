from os import name
from flask import Flask, request
from flask import render_template
from Predictor import Predictor


pred = Predictor()
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/answer", methods=["GET", "POST"])
def answer():
    userName = request.form['name']
    userResp = request.form['jobpost']
    ans = pred.predictor(userResp)

    return render_template("answer.html", name = userName, result = ans)
if __name__ == "__main__":
    app.run(debug=True)