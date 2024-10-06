from flask import Flask, render_template, request, redirect, url_for, session
from bidict import bidict
from random import choice
import numpy as np
from tensorflow import keras


#ascii table encoder
ENCODER = bidict({
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
    'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
    'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18,
    'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24,
    'Y': 25, 'Z': 26
})

app = Flask(__name__)
app.secret_key = "ai"


# main page
@app.route('/')
def index():
    session.clear()
    return render_template("index.html")


@app.route("/add-data", methods=["GET"])
def add_data_get():
    message = session.get("message", "")


    # get letters by its number in data

    labels = np.load('data/labels.npy')
    count = {k: 0 for k in ENCODER.keys()}
    for label in labels:
        count[label] += 1
    count = sorted(count.items(), key=lambda x: x[1])
    letter = count[0][0]


    # letter = choice(list(ENCODER.keys()))

    return render_template("addData.html", letter=letter, message = message)

@app.route("/add-data", methods=["POST"])
def add_data_post():

    # save letter
    label = request.form["letter"]
    labels = np.load("data/labels.npy")
    labels = np.append(labels, label)
    np.save("data/labels.npy", labels)

    # save letters image as array
    pixels = request.form["pixels"]
    pixels = pixels.split(",")
    img = np.array(pixels).astype(float).reshape(1,50,50)
    imgs = np.load("data/images.npy")
    imgs = np.vstack([imgs, img])
    np.save("data/images.npy", img)

    print(len(pixels))

    session["message"] = f'"{label}" added to dataset'

    return redirect(url_for("add_data_get"))

@app.route("/practice", methods=["GET"])
def practice_get():
    letter = choice(list(ENCODER.keys()))
    return render_template("practice.html", letter=letter)

@app.route("/practice", methods=["POST"])
def practice_post():
    letter = request.form['letter']

    pixels = request.form['pixels']
    pixels = pixels.split(',')
    img = np.array(pixels).astype(float).reshape(1, 50, 50, 1)

    model = keras.models.load_model('letter.model')

    pred_letter = np.argmax(model.predict(img), axis=-1)
    pred_letter = ENCODER.inverse[pred_letter[0]]

    correct = 'yes' if pred_letter == letter else 'no'
    letter = choice(list(ENCODER.keys()))

    return render_template("practice.html", letter=letter, correct=correct)


if __name__== "__main__":
    app.run(debug=True)