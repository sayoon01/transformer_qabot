# tokenize.py
import re

def tokenize(text):
    # 구두점도 분리해서 단어 단위로
    text = text.lower()
    tokens = re.findall(r"\w+|[^\s\w]", text)
    return tokens
