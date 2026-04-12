from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

reviews = ["good movie", "bad film", "awesome acting", "worst story", "great direction"]
labels = [1, 0, 1, 0, 1]  # 1=positive, 0=negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

model = MultinomialNB()
model.fit(X, labels)

test = ["awesome movie"]
test_vec = vectorizer.transform(test)

print("Prediction:", model.predict(test_vec))