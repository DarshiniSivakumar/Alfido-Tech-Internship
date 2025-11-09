import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import zipfile
zip_path=r"C:\Users\Darshini\Downloads\archive (6).zip"
extract_to=r"C:\Users\Darshini\Downloads\titanic_data"
with zipfile.ZipFile(zip_path,'r')as zp:
    zp.extractall(extract_to)
print("File extracted successfully")
csv_path=extract_to+r"\titanic.csv"
df=pd.read_csv(csv_path)
print("Missing values:",df.isnull().sum())
df['Age'].fillna(df['Age'].median(),inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)
data.drop('Cabin',axis=1,inplace=True)
data.drop(['Name','Ticket',PassengerId'],axis=1,inplace=True)
le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])
df['Embarked']=le.fit_transform(df['Embarked'])
X=df.drop('Survived',axis=1)
y=df['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
Lr_model=LogisticRegression()
lr_model.fit(X_train,y_train)
y_pred=lr_model.predict(X_test)
print("ACCURACY SCORE:",accuracy_scor(y_test,y_pred))
print("\nCLASSIFICATION REPORT:",classification_report(y_test,y_pred))
print("\nCONFUSION MATRIX:",confusion_matrix(y_test,y_pred))

