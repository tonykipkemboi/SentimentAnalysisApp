# Import dependencies
import base64
from io import BytesIO
from xlsxwriter import Workbook

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


@st.cache  # this function will be cached
def sentiment_score(review):
    """
    :param review:@review
    :type review:string
    :return:sentiment score
    :rtype:int
    """
    tok = tokenizer.encode(review, return_tensors='pt')
    res = model(tok)
    return int(torch.argmax(res.logits)) + 1


def to_excel(dfs):
    """
    :param dfs:dataframe
    :type dfs:dataframe
    :return:df
    :rtype:df
    """
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(dfm):
    """
    :param dfm:df
    :type dfm:df
    :return:xlsx
    :rtype:excel file
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octect-stream;base64,{b64.decode()}" download="extract.xlsx">Download Excel ' \
           f'File</a> '


if __name__ == '__main__':
    st.sidebar.image("assets/rev.png", use_column_width=True)
    st.header('Yelp Reviews Sentiment Analysis WebApp 👨‍💻')
    st.caption('The reviews and sentiment scores will be displayed below')
    yelp = "https://www.yelp.com/"
    st.sidebar.write("Link to Yelp: " + yelp)
    with st.sidebar.form(key='my_form'):
        path = st.text_input('Enter Yelp URL to analyze reviews', key='url')
        store_url = st.session_state.url
        submit_button = st.form_submit_button(label='Submit')

    # path = st.text_input('Yelp review site URL')
    try:
        r = requests.get(path)
        soup = BeautifulSoup(r.text, 'html.parser')
        regex = re.compile('.*comment.*')
        results = soup.find_all('p', {'class': regex})
        reviews = [result.text for result in results]

        df = pd.DataFrame(np.array(reviews), columns=['CUSTOMER REVIEW'])
        df['SENTIMENT SCORE'] = df['CUSTOMER REVIEW'].apply(lambda x: sentiment_score(x[:512]))
        score = df['SENTIMENT SCORE'].mean()
        num_rev = df['CUSTOMER REVIEW'].count()
        df.set_index('SENTIMENT SCORE')

        # display sentiment table page_bg_img = ''' <style> .stApp { background-image: url(
        # "https://mir-s3-cdn-cf.behance.net/project_modules/2800_opt_1/551074124683183.61095df8205df.jpg");
        # background-size: cover; } </style> ''' st.markdown(page_bg_img, unsafe_allow_html=True)

        st.write('Establishment Site URL: ', store_url)
        st.table(df)
        st.subheader('Sentiment Analysis Stats:')
        st.write('Number of Reviews: ', num_rev)
        st.write('Mean Sentiment Score: ', score, ' out of 5!')
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)
    except:
        pass

    
