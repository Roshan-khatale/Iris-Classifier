import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, "models/iris_model.pkl")
print("Model saved.")