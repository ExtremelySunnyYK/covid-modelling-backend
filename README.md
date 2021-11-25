# covid-modelling-backend

linear regression model from scratch deployed on flask

### URL

`https://covid-backend-modelling.herokuapp.com/`

## setup

Install python 3.8.5

Create virtual environment and install dependencies

Linux:

```bash
virtualenv env --python=python3.8

source env/bin/activate
pip install -r requirements.txt
```

Windows

```bash
virtualenv env --python=python3.8
env\Scripts\activate
pip install -r requirements.txt
```

## Run
```
export FLASK_APP=app.py
export FLASK_ENV=development
python wsgi.py or flask run
```

## Testing

`python -m pytest -vv |tee test.log`

<b>Predictor Endpoint</b> for  API to use our model

```
https://covid-backend-modelling.herokuapp.com/?data=[[-0.16797556263689653,%200.14091652326043888],%20[0.9056682418813299,%200.7959533187314604],%20[0.16237637721486545,%200.4285464387923033],%20[-0.9938554122663015,%20-0.8397883780364044],%20[0.9469622343628001,%200.7970117308312237],%20[1.2773141742145622,%200.9772745216071946],%20[1.2360201817330918,%200.9449288812826357],%20[-0.08538757767395605,%200.22898118243888696],%20[-1.3242073521180635,%20-1.5759785724985487],%20[-0.49832750248865854,%20-0.19130577270865506],%20[-1.6958532844512957,%20-2.2714485264004516],%20[0.49272831706662745,%200.6973901163914364],%20[1.6489601065477943,%201.1935311839518792],%20[-1.0351494047477718,%20-0.9083205606925213],%20[-1.3655013445995337,%20-1.678276795401853],%20[1.4837841366219133,%201.1105832720005908],%20[-0.6635034724145396,%20-0.3850600081911466]]
```

#### Debugging

`heroku logs --tail -a covid-backend-modelling`

`export FLASK_ENV=development`
