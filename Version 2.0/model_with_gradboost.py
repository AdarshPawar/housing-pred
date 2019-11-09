# Load libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot as plt


col_names = ['tot_people', 'food', 'amt', 'vehicle', 'distance', 'label']
house = pd.read_csv("data_set_formatted.csv", header = None, names = col_names)
house.head()


feature_cols = ['tot_people', 'food', 'amt', 'vehicle', 'distance']
X = house[feature_cols]
y = house.label

X_train = X[0:200]
X_test = X[201:]
y_train = y[0:200]

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


state = 12
test_size = 0.30

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=test_size, random_state=state)


gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=2, max_depth=2, random_state=0)
gb_clf.fit(X_train, y_train)
predictions = gb_clf.predict(X_test)


print("Learning rate: 0.05")
print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))
