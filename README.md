# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Read the data frame using pandas.
3.Get the information regarding the null values present in the dataframe.
4.Split the data into training and testing sets.
5.Convert the text data into a numerical representation using CountVectorizer.
6.Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7.Finally, evaluate the accuracy of the model.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Daisy R
RegisterNumber:212223220016  
*/
```
```
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
```
## Output:
## Result Output
![282257583-78ccb346-ca7c-4a33-ad4c-e3355e1fddc6](https://github.com/DaisyRavi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151394386/d9448238-e689-45df-9f15-8166741072ee)

## data.head()
![282257589-139f19db-04ee-4e44-b04a-5f231988b90b](https://github.com/DaisyRavi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151394386/e584fcd5-2ef9-4fd4-a57e-5cc1369d522e)

## data.info()
![282257589-139f19db-04ee-4e44-b04a-5f231988b90b](https://github.com/DaisyRavi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151394386/27e545db-febd-467f-9ca3-3151f8548838)

## data.isnull().sum()
![282257608-2eba109c-0bdd-468a-8bcf-d9258c23f8ef](https://github.com/DaisyRavi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151394386/2fe8b559-6efd-40ff-801c-d75a9197cce2)


## Y_prediction Value
![282257837-6a911f5c-1e40-4047-9371-07a94f012cef](https://github.com/DaisyRavi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151394386/c5b8927f-6f8e-4528-aed8-7bb311aca973)


## Accuracy Value
![282257853-bad89364-aef2-4652-806d-09c5760c041e](https://github.com/DaisyRavi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151394386/4b26c0c7-75bb-41b5-9b46-ac38be5fac37)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
