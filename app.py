from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_fun = [[int(x) for x in request.form.values()]]
    prediction = model.predict(int_fun)

    output = round(prediction[0], 2)

    return render_template('index.html', pretxt="The value on that year is Rs.{}".format(output))



