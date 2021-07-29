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

r = requests.get('https://www.yelp.com/biz/swahili-village-beltsville-6')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class': regex})
reviews = [result.text for result in results]

df = pd.DataFrame(np.array(reviews), columns=['CUSTOMER REVIEW'])


def sentiment_score(review):
    tok = tokenizer.encode(review, return_tensors='pt')
    res = model(tok)
    return int(torch.argmax(res.logits)) + 1


# Press the green button in the gutter to run the script
if __name__ == '__main__':
    st.write("""
    # Sentiment Analysis Web Application
    *The star ratings are out of a possible 5 star where 1 is most negative and 5 as most positive review*
    """)
    df['STAR RATING'] = df['CUSTOMER REVIEW'].apply(lambda x: sentiment_score(x[:512]))
    df.set_index('STAR RATING')
    st.table(df)
