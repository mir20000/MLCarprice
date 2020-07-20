
from flask import Flask, jsonify, request, render_template
import pickle

app = Flask(__name__)
reg = pickle.load(open('mlmodel.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    milage = int(request.form.get('milage'))
    age = int(request.form.get('age'))
    carnl = request.form.get('car')

    car = list(carnl)
    int_car = [int(x) for x in car]
    car = int_car

    feature_list = [[milage, age] + car]

    prediction = reg.predict(feature_list)

    output = round(prediction[0], 2)

    return render_template('index.html', pretxt="The value on that year is Rs.{}".format(output))


if __name__ == "__main__":
    app.run()



