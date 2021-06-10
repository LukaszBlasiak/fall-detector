from sklearn import datasets, svm, metrics
from dataset_util import load_dataset
from sklearn.externals import joblib

# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)
X_train, X_test, y_train, y_test = load_dataset(test_split_ratio=0.3, shuffle=True)

#Create a svm Classifier
# clf = svm.SVC(kernel='linear') # Linear Kernel
clf = svm.SVC(kernel='linear', gamma=0.01) # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train.ravel())

joblib.dump(clf, 'models/model_svm.pkl')

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test.ravel(), y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
