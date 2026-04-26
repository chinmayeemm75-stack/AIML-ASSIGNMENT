import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')

text = "I am learning NLP and it is very exciting!!!"

# Lowercase
text = text.lower()

# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# Tokenize
words = text.split()

# Remove stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if w not in stop_words]

# Stemming
ps = PorterStemmer()
words = [ps.stem(w) for w in words]

print("Processed Text:", words)