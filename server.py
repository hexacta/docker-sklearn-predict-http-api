from bottle import route, run, request, response
from sklearn.externals import joblib
import json
import pandas

model = joblib.load('model.pkl')


@route('/predict')
def predict():
    response.content_type = 'application/json'
    try:
        body = request.json
        body = body if isinstance(body, list) else [body]
        X = pandas.DataFrame.from_dict(body)
        result = model.predict(X)
        return json.dumps(result.tolist())
    except Exception as error:
        response.status = 500
        return json.dumps({'error': str(error)})


@route('/predictproba')
def predictproba():
    response.content_type = 'application/json'
    try:
        body = request.json
        body = body if isinstance(body, list) else [body]
        X = pandas.DataFrame.from_dict(body)
        result = pandas.DataFrame(model.predict_proba(X),
                                  columns=model.classes_)
        return result.to_json(orient="records")
    except Exception as error:
        response.status = 500
        return json.dumps({'error': str(error)})


@route('/featurenames')
def featurenames():
    response.content_type = 'application/json'
    try:
        result = model.get_booster().feature_names
        return json.dumps(result)
    except Exception as error:
        response.status = 500
        return json.dumps({'error': str(error)})


#run(host='localhost', port=8080, debug=True, reloader=True)
run(host='0.0.0.0', port=8080)

# curl -H "Content-Type: application/json" -X GET -d '[{"sepal length (cm)":4.4}]' http://localhost:8080/predictproba
