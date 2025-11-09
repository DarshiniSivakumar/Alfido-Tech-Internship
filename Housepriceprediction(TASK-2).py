import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt
from sklearn.preprocessing import LabelEncoder
zip_path=r"C:\Users\Darshini\Downloads\archive (5).zip"
extract_to=r"C:\Users\Darshini\Downloads\house_data_folder"
with zipfile.ZipFile
(zip_path,'r') as zp:
     zp.extractall(extract_to)
print("File extracted successfully")
file_path=extract_to+r"\data.csv"
df=pd.read_csv(file_path)
print("File loaded successfully")
if 'date' in df.columns:
     df.drop('date',axis=1)
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"Encoding column: {col}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
X=df.drop('price',axis=1)
y=df['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("ROOT MEAN SQUARED ERROR:",sqrt(mean_squared_error(y_test,y_pred)))
print("MEAN ABOSLUTE ERROR:",mean_absolute_error(y_test,y_pred))
print("R^2 SCORE:",r2_score(y_test,y_pred))



