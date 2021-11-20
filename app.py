import numpy as np
from flask import Flask, request, jsonify, render_template
import json
from model import Model


def create_app():
    app = Flask(__name__)

    return app


app = create_app()
model = Model()


@app.route('/')
def home():
    return render_template('index.html')


@ app.route('/predict', methods=['GET'])
def predict():
    if request.method == 'GET':
        data = request.args.get('data')
        app.logger.info(data)
        lst = json.loads(data)
        prediction = model.predict(lst)
        # convert numpy array to list
        prediction = prediction.tolist()
        return jsonify(prediction)

    elif request.method == 'POST':
        data = request.get_json()
        app.logger.info(data)
        lst = json.loads(data)
        prediction = model.predict(lst)
        # convert numpy array to list
        prediction = prediction.tolist()
        return jsonify(prediction)


# @ app.route('/view', methods=['POST', 'GET'])
# def view():

#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)

#     output = round(prediction[0], 2)

#     return render_template('index.html', prediction_text='STI should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
