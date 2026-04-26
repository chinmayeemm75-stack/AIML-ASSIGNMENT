import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.2)

accuracies = []

for k in range(1,16):
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)
    accuracies.append(acc)

# Plot
plt.plot(range(1,16), accuracies)
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.show()

print("Best K:", accuracies.index(max(accuracies))+1)