# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier                                                             # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split                                                        # Import train_test_split function
from sklearn import metrics                                                                                 #Import scikit-learn metrics module for accuracy calculation

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus


col_names = ['tot_people', 'food', 'amt', 'vehicle', 'distance', 'label']
house = pd.read_csv("data_set_formatted.csv", header = None, names = col_names)
house.head()


feature_cols = ['tot_people', 'food', 'amt', 'vehicle', 'distance']
X = house[feature_cols]
y = house.label


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1','2','3','4','5','6','7','8','9','10',
                                                                                  '11','12','13','14','15'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('housing.png')
Image(graph.create_png())