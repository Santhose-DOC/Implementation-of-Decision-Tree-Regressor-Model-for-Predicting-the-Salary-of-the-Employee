# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```python
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
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

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: Santhose Arockiaraj J

RegisterNumber:  212224230248


## Output:

### Data Head :

![Screenshot 2025-05-12 111402](https://github.com/user-attachments/assets/bf71a456-01f1-465c-93ab-c7b7c29f8208)

### Data Info :

![Screenshot 2025-05-12 111635](https://github.com/user-attachments/assets/f3228236-465a-4374-94be-89b35464dc83)

### isnull().sum() :

![Screenshot 2025-05-12 111703](https://github.com/user-attachments/assets/abe877d3-93ac-4e82-ab89-6b9d10545c03)

### Data Head for salary :

![Screenshot 2025-05-12 111726](https://github.com/user-attachments/assets/7cf05adb-822b-4ec0-a42c-19264c2221ae)

### Mean Squared Error :

![Screenshot 2025-05-12 132306](https://github.com/user-attachments/assets/d61f864f-58e1-4d84-960b-275503b585e1)

### r2 Value:

![Screenshot 2025-05-12 132327](https://github.com/user-attachments/assets/c7fc1880-6890-48b3-be11-d99eb18ff9be)

### Data prediction :

![Screenshot 2025-05-12 132419](https://github.com/user-attachments/assets/a322cbb5-bd3b-4e62-960a-b7481367e96f)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
