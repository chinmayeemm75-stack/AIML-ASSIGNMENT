from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

data = load_breast_cancer()
X = data.data
y = data.target

model = LogisticRegression(max_iter=5000)
model.fit(X,y)

probs = model.predict_proba(X)[:,1]

thresholds = [0.3,0.5,0.7]

for t in thresholds:
    preds = (probs >= t).astype(int)
    
    print(f"\nThreshold: {t}")
    print("Accuracy:", accuracy_score(y,preds))
    print("Precision:", precision_score(y,preds))
    print("Recall:", recall_score(y,preds))