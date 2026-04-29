from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
emails = [
    "Win a free lottery now",
    "Congratulations you have won a prize",
    "Claim your free gift card",
    "Meeting at 10 am tomorrow",
    "Let's complete the project today",
    "Are you available for discussion"
]

labels = [1, 1, 1, 0, 0, 0]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Input from user
new_email = input("Enter an email: ")

# Transform input
new_email_vec = vectorizer.transform([new_email])

# Prediction
result = model.predict(new_email_vec)

# Output
if result[0] == 1:
    print("This is a SPAM email")
else:
    print("This is NOT a spam email")