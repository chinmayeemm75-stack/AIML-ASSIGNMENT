# Spam Email Classifier

import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "text": [
        "Win money now!!!",
        "Call me later",
        "Congratulations, you won a prize",
        "Let's meet tomorrow",
        "Free entry in a contest",
        "How are you doing?"
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1=Spam, 0=Ham
}

df = pd.DataFrame(data)

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

test_email = ["You have won cash prize"]
test_vec = vectorizer.transform(test_email)
prediction = model.predict(test_vec)

print("Spam" if prediction[0] == 1 else "Not Spam")