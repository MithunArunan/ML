# Import dataset, classifier and metrics
from sklearn import datasets, svm, metrics
import pickle
from sklearn.externals import joblib

# Loading the 0-9 labelled digits dataset
digits_dataset = datasets.load_digits()

# Size of the dataset - row count + column count
#print(type(digits_dataset.target))
print(str(len(digits_dataset.data))+"*"+str(len(digits_dataset.data[0])))
print(str(len(digits_dataset.target)))

# Creating an estimator instance of type Support Vector Classifier
classifier = svm.SVC(gamma=0.001, C=100.)

# Fitting the model - Training and learning
classifier.fit(digits_dataset.data[:-1], digits_dataset.target[:-1])

# Model persitence Saving the model as pickle
classifier_str = pickle.dumps(classifier)
joblib.dump(classifier, 'digits_classifier.pkl')

# Prediction
# print(classifier.predict(digits_dataset.data[-1:]))
classifier_pkl = joblib.load('digits_classifier.pkl')
print(classifier_pkl.predict(digits_dataset.data[-1:]))
