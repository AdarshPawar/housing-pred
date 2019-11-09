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
house = pd.read_csv("data_set_formatted.csv", header = None, names = col_names)
house.head()


feature_cols = ['tot_people', 'food', 'amt', 'vehicle', 'distance']
X = house[feature_cols]
y = house.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# # Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)

clf = clf.fit(X_train,y_train)

prediction = clf.predict(arr)
print(prediction)

if(prediction[0] == 0):
    print ('Udupi PG\nSingle Sharing\nAmount - 8000\nDistance from college - 1km\nFood - Provided')
elif(prediction[0] == 1):
    print('Udupi PG\n2 Sharing\nAmount - 6000\nDistance from college - 1km\nFood - Provided')
elif(prediction[0] == 2):
    print('Spandana PG\nSingle Sharing\nAmount - 7000\nDistance from college - 3km\nFood - Provided')
elif(prediction[0] == 3):
    print('Spandana PG\n2 Sharing\nAmount - 5000\nDistance from college - 3km\nFood - Provided')
elif(prediction[0] == 4):
    print('Spandana PG\n3 Sharing\nAmount - 5000\nDistance from college - 3km\nFood - Provided')
elif(prediction[0] == 5):
    print('Suryodhaya PG\n3 Sharing\nAmount - 4000\nDistance from college - 5km\nFood - Provided')
elif(prediction[0] == 6):
    print ('Udupi Rooms\nSingle Sharing Room\nAmount - 6500\nDistance from college - 1km\nFood - Not Provided')
elif(prediction[0] == 7):
    print ('Udupi Rooms\n2 Sharing Room\nAmount - 4500\nDistance from college - 1km\nFood - Not Provided')
elif(prediction[0] == 8):
    print('Spandana Rooms\nSingle Sharing Room\nAmount - 5500\nDistance from college - 3km\nFood - Not Provided')
elif(prediction[0] == 9):
    print('Spandana Rooms\n3 Sharing Room\nAmount - 3500\nDistance from college - 3km\nFood - Not Provided')
elif(prediction[0] == 10):
    print('Suryodhaya Rooms\n2 Sharing Room\nAmount - 3500\nDistance from college - 5km\nFood - Not Provided')
elif(prediction[0] == 11):
    print('Suryodhaya Rooms\n3 Sharing Room\nAmount - 2500\nDistance from college - 5km\nFood - Not Provided')
elif(prediction[0] == 12):
    print('Udupi Houses\nHouse\nAmount - 3000\nDistance from college - 1km\nFood - Not Provided')
elif(prediction[0] == 13):
    print('Spandana Houses\nHouse\nAmount - 2500\nDistance from college - 3km\nFood - Not Provided')
elif(prediction[0] == 14):
    print('Suryodhaya Houses\nHouse\nAmount - 1500\nDistance from college - 5km\nFood - Not Provided')
elif(prediction[0] == 15):
    print ('Hostel\n3 Sharing\nAmount - 5000\nDistance from college - 1km\nFood - Provided')