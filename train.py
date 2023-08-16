from sklearn import datasets
from sklearn import svm

import bentoml

#Load dataset

iris = datasets.load_iris()
X,y = iris.data, iris.target 

#training a model
clf = svm.SVC(gamma='scale')
clf.fit(X,y)

#Save model to BentoML local model store
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(f"Model Saved: {saved_model}")