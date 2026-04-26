import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    "income":[50000,60000,40000,80000,30000,90000],
    "credit_score":[700,750,600,800,550,850],
    "loan_amount":[20000,25000,15000,30000,10000,35000],
    "employment_years":[5,6,3,8,2,10],
    "approved":[1,1,0,1,0,1]
}

df = pd.DataFrame(data)

X = df.drop("approved",axis=1)
y = df["approved"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LogisticRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,pred))

# Predict new customer
new = [[70000,720,20000,5]]
print("Prediction:", model.predict(new))