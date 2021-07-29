# Import dependencies
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import streamlit as st

# Instantiate model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Encode and calculate sentiment
tokens = tokenizer.encode('love', return_tensors='pt')
result = model(tokens)

r = requests.get('https://www.yelp.com/biz/embassy-of-kenya-washington')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class': regex})
reviews = [result.text for result in results]

df = pd.DataFrame(np.array(reviews), columns=['review'])


def sentiment_score(review):
    tok = tokenizer.encode(review, return_tensors='pt')
    res = model(tok)
    return int(torch.argmax(res.logits)) + 1


# Press the green button in the gutter to run the script
if __name__ == '__main__':
    df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))
    st.table(df)
