# docker-sklearn-predict-http-api
A docker image that provides a web api to a sklearn model prediction methods

## Usage

Build your model:
```py
from sklearn import svm
from sklearn import datasets
import pandas as pd
clf = svm.SVC()
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
```bash
docker build -t iris-svc .
docker run -p 4000:8080 iris-svc
```

Make a request to the API:
```bash
curl -H "Content-Type: application/json" -X GET -d '[{"sepal length (cm)":4.4}]' http://localhost:4000/predictproba
```


## API
TODO

## License

MIT © [Hexacta](http://www.hexacta.com)