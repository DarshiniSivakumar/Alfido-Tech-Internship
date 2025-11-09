import zipfile
import pandas as pd
import numpy as np
import sklearn.model_selection import train_test_split
import sklearn.linear_model import LinearRegression
import sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
zip_path=r"C:\Users\Darshini\Downloads\archive (5).zip"
extract_to="C:\Users\Darshini\Downloads\house_data_folder"
with zipfile.ZipFile
(zip_path,'r') as zp:
     zp.extractall(zip_path)
print("File extracted successfully")
file_path=extract_to+r"\data.csv"
df=pd.read_csv(file_path)
print("File loaded successfully")
if 'date' in df.columns:
     df.drop('date',axis=1)
X=df.drop('price',axis=1)
y=df['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("ROOT MEAN SQUARED ERROR:",sqrt(mean_squared_error(y_test,y_pred)))
print("MEAN ABOSLUTE ERROR:",mean_absolute_error(y_test,y_pred))
print("R^2 SCORE:",r2_score(y_test,y_pred))


