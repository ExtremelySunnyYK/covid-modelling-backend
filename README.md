# covid-modelling-backend

linear regression model from scratch deployed on flask

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

`python app.py`

## Testing

`python -m pytest -vv |tee test.log`

### Endpoint

`https://covid-backend-modelling.herokuapp.com/`

#### Debugging

`heroku logs --tail -a covid-backend-modelling`
