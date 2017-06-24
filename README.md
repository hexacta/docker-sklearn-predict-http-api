# docker-sklearn-predict-http-api
A docker image that provides a web api to a sklearn model prediction methods

## Usage

Build your model:
```py
from sklearn import svm
from sklearn import datasets
import pandas as pd
clf = svm.SVC(probability=True)
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
X = X[["sepal length (cm)"]]
y = iris.target
clf.fit(X, y)  
```

Save it:
```py
from sklearn.externals import joblib
joblib.dump(clf, 'iris-svc.pkl')
```

Create a `Dockerfile` from "hexacta/sklearn-predict-http-api" and copy your model:
```Dockerfile
FROM hexacta/sklearn-predict-http-api:latest
COPY iris-svc.pkl /usr/src/app/model.pkl
```

Build and run the image using `docker`:
```console
$ docker build -t iris-svc .
$ docker run -d -p 4000:8080 iris-svc
```

Make requests to the API:
```console
$ curl -H "Content-Type: application/json" -X GET -d '{"sepal length (cm)":4.4}' http://localhost:4000/predictproba
  [{"0":0.8284069169,"1":0.1077571623,"2":0.0638359208}]
$ curl -H "Content-Type: application/json" -X GET -d '[{"sepal length (cm)":4.4}, {"sepal length (cm)":15}]' http://localhost:4000/predict
  [0, 2]
```


## API
TODO

## License

MIT © [Hexacta](http://www.hexacta.com)