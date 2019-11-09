# Load libraries
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier                                                             # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split                                                        # Import train_test_split function
from sklearn import metrics                                                                                 #Import scikit-learn metrics module for accuracy calculation

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

arr=[]

totalPeople=int(input("How many people would you like to share the room with :\n"))
food=input("Would you like to have a room along with food facilities?(Answer yes or no)\n")
if food=='yes':
    food=1
else:
    food=0
amount=int(input("How much would you like to spend on the room?\n"))
vehicle=input("Do you own a vehicle?(Answer yes or no)\n")
if vehicle=='yes':
    vehicle=1
else:
    vehicle=0
distance=int(input("How far would you like to have your room from the college?\n"))
arr=np.array([totalPeople,food,amount,vehicle,distance])
arr=np.array(arr)

arr=arr.reshape(1, -1)

col_names = ['tot_people', 'food', 'amt', 'vehicle', 'distance', 'label']
house = pd.read_csv("Simple_Label_Formatted.csv", header = None, names = col_names)
house.head()


feature_cols = ['tot_people', 'food', 'amt', 'vehicle', 'distance']
X = house[feature_cols]
y = house.label


# # Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)

clf = clf.fit(X,y)

pred = clf.predict(arr)

if pred[0]==0:
    print("Hostel")
elif pred[0]=='1':
    print("PG")
elif pred[0]=='2':
    print("Room")
else:
    print("House")