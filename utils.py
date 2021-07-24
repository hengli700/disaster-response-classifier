"""
Utility module
"""

import re
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords


def tokenize(text):
    """
    Transform input text into clean tokens with special character removed.
    :param text: string - input text to be tokenized
    :return: list - clean_tokens
    """
    # replace any non alphanumeric character with space, and lowercase the text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token)
        clean_tokens.append(clean_token)
    return clean_tokens
