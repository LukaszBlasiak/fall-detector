from sklearn.ensemble import RandomForestClassifier
from dataset_util import load_dataset
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.externals import joblib

# Load data and store it into pandas DataFrame objects
X_train, X_test, y_train, y_test = load_dataset(test_split_ratio=0.3, shuffle=True)

# Defining and fitting a DecisionTreeClassifier instance
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

joblib.dump(clf, 'models/model__random_forest.pkl')

y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test.ravel(), y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
