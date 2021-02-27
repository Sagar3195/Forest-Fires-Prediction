import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

###Loading dataset
df = pd.read_csv("Forest_fire.csv")
print(df.head())

print(df.shape)
###Now we check missing values in dataset
print(df.isnull().sum())

##We can see that there are some missing values in dataset
#Now we split dataset into independent variable and dependent variable
data = np.array(df)

X = data[:,1:-1]
y = data[:, -1]

print(X.shape, y.shape)

##we convert object data type into int data types using astype method
X = X.astype("int")
y = y.astype("int")

##Now we split dataset into training data  and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

print(x_train.shape, y_train.shape)
# #### RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators= 10)
forest.fit(x_train, y_train)

y_pred = forest.predict(x_test)

print(y_pred)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
##Let's check performance of the model
forest_accuracy = accuracy_score(y_test, y_pred)
print(forest_accuracy)

forest_cm = confusion_matrix(y_test, y_pred)
print(forest_cm)

print(classification_report(y_test, y_pred))

import pickle
###Open a file, where we want 
file = open("forest_fire_model.pkl", 'wb')
##dump information to that file
pickle.dump(forest, file)
