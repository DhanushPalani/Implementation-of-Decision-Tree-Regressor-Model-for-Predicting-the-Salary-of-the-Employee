# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the
6. Required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: UDHAYANITHI M
RegisterNumber: 212222220054 
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv("/content/Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
data["Position"].value_counts()
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x = data[["Position","Level"]]
y=data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

# Make predictions on the test set
y_pred = dt.predict(x_test)
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20,8))

plot_tree(dt,feature_names=x.columns, filled=True)
plt.show()
```

## Output:
![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/127933352/0035654d-678e-4cfd-9cea-8e4a7fbedca1)

![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/127933352/14a1b5ba-2cc3-4511-9481-0978ac2ae39b)
![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/127933352/cc77766c-2538-45ee-b352-b1400850e987)

![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/127933352/67116f0e-559e-4f48-a381-1ef1831ad86c)
![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/127933352/46f393ca-3a93-4bf4-9d9c-96de10819327)
![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/127933352/9b0445fb-b9dd-48fb-b46e-baa8695e3676)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
