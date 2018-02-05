# data preprocessing
import pandas as pd

train = pd.read_csv('train.csv')
x_train = train.iloc[ :, :-1].values
y_train = train.iloc[:, -1]

train.isnull().values.any()

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)

# encoding categorical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# implementing random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='gini', max_features=100, random_state=0)
classifier.fit(x_train, y_train)

# test data
# importing
test = pd.read_csv('test.csv')
x_test = test.iloc[ :, :-1].values
y_test = test.iloc[:, -1]

test.isnull().values.any()

# feature scaling
x_test = sc_x.fit_transform(x_test)

# encoding categorical values
y_test = le.fit_transform(y_test)

# predicting the result
y_pred = classifier.predict(x_test)

# checking accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# k-fold cross validation score
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv=10)
print("mean of the accuracies is ",accuracies.mean())
print("standard deviation of amodel is", accuracies.std())

# checking accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
