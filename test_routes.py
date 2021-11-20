from werkzeug.wrappers import response
import pytest
from pytest import fixture
from app import app
import numpy as np
from flask import request
import json


@pytest.fixture
def client():
    # app = create_app()
    app.config['TESTING'] = True

    with app.app_context():
        with app.test_client() as client:
            yield client


def test_empty_page(client):
    """Sample test"""
    response = client.get('/')
    assert response.status_code == 200
    assert response.data == b'Hello, World!'


def test_predict(client):
    """ Making a get request to predict endpoint with list of features"""
    data = [[-0.16797556263689653, 0.14091652326043888],
            [0.9056682418813299, 0.7959533187314604],
            [0.16237637721486545, 0.4285464387923033],
            [-0.9938554122663015, -0.8397883780364044],
            [0.9469622343628001, 0.7970117308312237],
            [1.2773141742145622, 0.9772745216071946],
            [1.2360201817330918, 0.9449288812826357],
            [-0.08538757767395605, 0.22898118243888696],
            [-1.3242073521180635, -1.5759785724985487],
            [-0.49832750248865854, -0.19130577270865506],
            [-1.6958532844512957, -2.2714485264004516],
            [0.49272831706662745, 0.6973901163914364],
            [1.6489601065477943, 1.1935311839518792],
            [-1.0351494047477718, -0.9083205606925213],
            [-1.3655013445995337, -1.678276795401853],
            [1.4837841366219133, 1.1105832720005908],
            [-0.6635034724145396, -0.3850600081911466]]

    string = json.dumps(data)

    # Query parameter : http://127.0.0.1:5000/predict?data + list of features
    # http://127.0.0.1:5000/predict?data=[[-0.16797556263689653,%200.14091652326043888],%20[0.9056682418813299,%200.7959533187314604],%20[0.16237637721486545,%200.4285464387923033],%20[-0.9938554122663015,%20-0.8397883780364044],%20[0.9469622343628001,%200.7970117308312237],%20[1.2773141742145622,%200.9772745216071946],%20[1.2360201817330918,%200.9449288812826357],%20[-0.08538757767395605,%200.22898118243888696],%20[-1.3242073521180635,%20-1.5759785724985487],%20[-0.49832750248865854,%20-0.19130577270865506],%20[-1.6958532844512957,%20-2.2714485264004516],%20[0.49272831706662745,%200.6973901163914364],%20[1.6489601065477943,%201.1935311839518792],%20[-1.0351494047477718,%20-0.9083205606925213],%20[-1.3655013445995337,%20-1.678276795401853],%20[1.4837841366219133,%201.1105832720005908],%20[-0.6635034724145396,%20-0.3850600081911466]]
    response = client.get(f'/predict/data={data}')

    print(response)

    assert response.status_code == 200
