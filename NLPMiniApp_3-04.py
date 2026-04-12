import string

# Sample chatbot + keyword extractor
stop_words = ["is", "am", "are", "the", "and", "a", "an"]

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return [word for word in words if word not in stop_words]

def chatbot_response(text):
    words = preprocess(text)
    
    if "hello" in words or "hi" in words:
        return "Hello! How can I help you?"
    elif "ai" in words:
        return "AI is transforming the world!"
    else:
        return "Sorry, I didn't understand."

# Test
user_input = "Hello, I am learning AI!"
print("Keywords:", preprocess(user_input))
print("Bot:", chatbot_response(user_input))