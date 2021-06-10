import pandas as pd
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

model = joblib.load("models/iris_model.pkl")
y_pred = model.predict(X)
print(classification_report(y, y_pred))