# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries from python.
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeRegressor from sklearn and apply to the model from the dataset.
5.Predict the values of the arrays.
6.Import metrics from sklearn and calculate the MSE and R2 of the model from the dataset.
7.Predict the values of array
8.Apply it to the new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: sekar M
RegisterNumber:  212225230257
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

```

## Output:

<img width="1036" height="322" alt="image" src="https://github.com/user-attachments/assets/6c87600b-7c93-48ad-a1cb-5ffa327dfbef" />
<img width="497" height="235" alt="image" src="https://github.com/user-attachments/assets/3ea65e15-0584-478a-b6cc-9874409e582e" />
<img width="857" height="158" alt="image" src="https://github.com/user-attachments/assets/1f9faa3b-43ba-41ad-a5e1-e03f82fce917" />
<img width="392" height="27" alt="image" src="https://github.com/user-attachments/assets/5dccea69-17dc-4b7f-8650-93fb0b48eb3f" />
<img width="237" height="33" alt="image" src="https://github.com/user-attachments/assets/0f8923b1-c34b-429a-9abd-85d695ee0879" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
