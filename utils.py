import re
from nltk import word_tokenize, WordNetLemmatizer


def tokenize(text):
    # replace any non alphanumeric character with space, and lowercase the text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token)
        clean_tokens.append(clean_token)
    return clean_tokens
