import pandas as pd
from sklearn.linear_model import LinearRegression

# Create dataset
data = {
    "engine": [1000,1200,1500,1800,2000],
    "mileage": [20,18,15,12,10],
    "age": [5,4,3,2,1],
    "price": [3,4,6,8,10]
}

df = pd.DataFrame(data)

# Features & target
X = df[["engine","mileage","age"]]
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X,y)

# Predict
prediction = model.predict([[1500,20,3]])
print("Predicted Price:", prediction[0])

# Coefficients
print("Coefficients:", model.coef_)

# Interpretation
features = ["engine","mileage","age"]
for f,c in zip(features, model.coef_):
    print(f"{f} impact: {c}")