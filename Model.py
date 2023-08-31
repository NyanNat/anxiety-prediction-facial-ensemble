import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


def fit_multiple_estimators(classifiers, X_list, y, sample_weights = None):

    # Convert the labels `y` using LabelEncoder, because the predict method is using index-based pointers
    # which will be converted back to original data later.
    le_ = LabelEncoder()
    le_.fit(y)
    transformed_y = le_.transform(y)

    # Fit all estimators with their respective feature arrays
    estimators_ = [clf.fit(X, y) if sample_weights is None else clf.fit(X, y, sample_weights) for clf, X in zip([clf for _, clf in classifiers], X_list)]

    return estimators_, le_


def predict_from_multiple_estimator(estimators, label_encoder, X_list, weights = [0.5, 0.25, 0.25]):

    # Predict 'soft' voting with probabilities

    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    pred2 = np.average(pred1, axis=0, weights=weights)
    pred = np.argmax(pred2, axis=1)

    # Convert integer predictions to original labels:
    return label_encoder.inverse_transform(pred)

def predict_output(input):
    X_input = input[3:16]
    X_input = np.array(X_input)

    X_input_hrv = X_input[[6,7]].reshape(1,-1)
    X_input_eye = X_input[[8]].reshape(1,-1)
    X_input_head = X_input[[0,1,2,3,4,5,9,10,11,12]].reshape(1,-1)
    X_input_list = [X_input_eye, X_input_hrv, X_input_head]

    y_pred = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_input_list)
    return y_pred

df = pd.read_csv('data.csv')
X = df.iloc[:, [3,4,5,6,7,8,9,10,11,12,13,14,15]].values
Y = df.iloc[:, 16].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

X_train_hrv = X_train_res[:,[6,7]]
X_train_eye = X_train_res[:,[8]]
X_train_head = X_train_res[:,[0,1,2,3,4,5,9,10,11,12]]
X_test_hrv = X_test[:,[6,7]]
X_test_eye = X_test[:,[8]]
X_test_head = X_test[:,[0,1,2,3,4,5,9,10,11,12]]

X_train_list = [X_train_eye, X_train_hrv, X_train_head]
X_test_list = [X_test_eye, X_test_hrv, X_test_head]

classifiers = [('ada', AdaBoostClassifier(estimator = RandomForestClassifier(), n_estimators = 13, random_state = 109)), ('svc1', SVC(kernel = 'rbf', random_state = 109, C = 2, probability=True)), ('svc2', SVC(kernel = 'rbf', random_state = 109, C = 2, probability=True))]

fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, y_train_res)