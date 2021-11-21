from flask import Flask, request, jsonify, render_template
import json
import numpy as np
from model import Model
import logging
import datetime as dt
import datetime as dt


def create_app():
    app = Flask(__name__)

    return app


app = create_app()
model = Model()
logging.basicConfig(filename='record.log', level=logging.DEBUG,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


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


@ app.route('/view', methods=['POST', 'GET'])
def view():
    form_values = [v for v in request.form.values()]
    # convert value to datetime
    date = form_values[0]
    date = dt.datetime.strptime(date, '%d/%m/%y')
    # convert datetime to ordinal
    form_values[0] = dt.datetime.toordinal(date)

    int_features = [int(x) for x in form_values]

    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    app.logger.info('%s prediction success', prediction)

    output = round(prediction[0, 0], 2)

    return render_template('index.html', prediction_text='STI should be $ {}'.format(output))
