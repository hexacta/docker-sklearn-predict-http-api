from bottle import route, run, request, response
from sklearn.externals import joblib
import json
import pandas

model = joblib.load('model.pkl')


@route('/predict')
def predict():
    response.content_type = 'application/json'
    body = request.json
    body = body if isinstance(body, list) else [body]
    X = pandas.DataFrame.from_dict(body)
    result = model.predict(X)
    return json.dumps(result.tolist())


@route('/predictproba')
def predictproba():
    response.content_type = 'application/json'
    body = request.json
    body = body if isinstance(body, list) else [body]
    X = pandas.DataFrame.from_dict(body)
    result = pandas.DataFrame(model.predict_proba(X),
                              columns=model.classes_)
    return result.to_json(orient="records")


@route('/featurenames')
def featurenames():
    response.content_type = 'application/json'
    result = model.get_booster().feature_names
    return json.dumps(result)


#run(host='localhost', port=8080, debug=True, reloader=True)
run(host='0.0.0.0', port=8080)

# curl -H "Content-Type: application/json" -X GET -d '[{"Age":1},{"Age":25}]' http://localhost:8080/predictproba
