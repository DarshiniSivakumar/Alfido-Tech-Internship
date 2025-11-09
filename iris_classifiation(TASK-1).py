import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisiticRegression
from sklearn.metrics import acuracy_score,classification_report
import zipfile
zip_path = r"C:\Users\Darshini\Downloads\archive (4).zip"
extract_to = r"C:\Users\Darshini\Downloads\iris_data"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
print("Files extracted to:", extract_to)
csv_path = extract_to + r"\IRIS.csv"  
df = pd.read_csv(csv_path)
print("Dataset loaded successfully!\n")
print(df.head())
df=pd.read_csv("Iris.csv")
X=df.drop("species",axis=1)
y=df["species"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
dt_model=DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train,y_train)
y_pred=dt_model.predict(X_test)
print("DECISION TREE CLASSIFIER")
print("ACCURACY:",accuracy_score(y_test,y_pred)
print("Classification:",classification_report(y_test,y_pred)
lr_model=LogisticRegression(max_iter=200)
lr_model.fit(X_train,y_train)
y_pred_lr=lr_model.predict(X_test)
print("Logistic Regression")
print("ACCURACY:",accuracy_score(y_test,y_pred)
print("Classification:",classification_report(y_test,y_pred)
