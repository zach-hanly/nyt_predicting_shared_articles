"""functions for #nyt_predicting_shared_articles 
"""

# import required packages
import os
import requests 
import re
import json

import pandas as pd
import math

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

"""

Table of Contents

    API request functions:
        get_api_key(path)
        get_articles(year_month)
        get_most_shared_articles(search_period)

    EDA funcitons:
        load_most_shared_eda(dir_path)
        clean_most_shared_eda(df_list)
        plot_most_shared(df_list)

    Data Cleaning funciton:
        cleaned_articles(archive)
        cleaned_shared_articles(most_shared_articles)
        load_most_shared(dir_path)

    Sampling funcitons:
        smote_data(X_train, y_train, sampling_strategy, random_state)

    NLP functions:
        get_wordnet_pos(treebank_tag)
        text_prep(text, sw)
        vectorize_feature(vectorizer, X_train, X_test)
        tokenize_vector(vectorizer, X_train, X_test)
        plot_top_words(vectorizer, X_train)
    
"""

"""
API request functions
""" 

#function to access private API key
def get_api_key(path):
    with open(path) as f:
        return json.load(f)

    
# function to get a list of all articles for provided months from API
def get_articles(year_month):
    articles_list = []
    
    # get API key from private folder in director out of repo 
    api_key = get_api_key("../.nyt_api.json")['api_key']
    
    # make API call for every month passed through 
    for date in year_month:
        year = str(date[0])
        month = str(date[1])
        url = f'https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={api_key}'
        response = requests.get(url)
        response_json = response.json()
        articles = response_json['response']['docs']
        articles_list.extend(articles)
        
    return articles_list 


# function to get a list of most shared articles for provided time period (in days)
def get_most_shared_articles(search_period):
    most_shared_articles = []
    period = str(search_period)
    
    # get API key from private folder in director out of repo
    api_key = get_api_key("../.nyt_api.json")['api_key']
    url = f'https://api.nytimes.com/svc/mostpopular/v2/shared/{period}/facebook.json?api-key={api_key}'
    response = requests.get(url)
    response_json = response.json()
    most_shared_articles = response_json['results']
    
    return most_shared_articles



"""
EDA funcitons
""" 

# function to load most shared articles for eda
def load_most_shared_eda(dir_path):
    
    directory = os.fsencode(dir_path)
    
    files = []
    most_shared_df = pd.DataFrame()
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        exclude = 'before'
        if filename.endswith('.csv') and exclude not in filename :
            files.append(filename)
            
    #read them into pandas
    df_list = [pd.read_csv(dir_path+'/'+file) for file in files]
    
    return df_list


# fucntion to clean most shared articles for eda
def clean_most_shared_eda(df_list):
    
    for df in df_list:
        df.date_published = df.date_published.apply(lambda x: pd.to_datetime(x).date())
        df.date_sourced = df.date_sourced.apply(lambda x: pd.to_datetime(x).date())
        df.set_index('date_sourced', inplace=True)
        df.sort_index(ascending=False, inplace=True)

    df_list.sort(key=lambda x: x.index[0])
    return df_list


# function to present and plot distribtuion of values in series
# automatically prints plot, summary df is returned to unpack and present 
def summerize_value_counts(series):
    
    # extract name of series
    series_name = series.name
    
    # make dataframe to display value count sum and percentage for series 
    series_count = series.value_counts().rename('sum')
    series_perc = series.value_counts(normalize=True).round(2).rename('percentage')
    series_values_df = pd.concat([series_count, series_perc], axis=1)
    
    # plot series distribution 
    plot = series_values_df['sum'].plot(kind='bar', title=f'Distribution of {series_name.title()} Column', 
                                 xlabel=f'{series_name.title()}', ylabel='Count');
    
    # rename df index to series name
    series_values_df.index.name = series_name
    series_values_df
    
    return series_values_df


# function to plot each days most shared articles by count of what day the articles 
# were origionally published on
def plot_most_shared(df_list):
    
    size = len(df_list)
    cols = round(math.sqrt(size))
    rows = cols
    while rows * cols < size:
        rows += 1
    f, ax_arr = plt.subplots(nrows=rows, ncols=cols, figsize=(20,20))
    plt.subplots_adjust(wspace=0.35, hspace=0.35)
    ax_arr = ax_arr.reshape(-1)
    for i in range(len(ax_arr)):
        if i >= size:
            ax_arr[i].axis('off')
            break

        df_list[i].groupby('date_published').count().plot(kind='bar', 
                                                          title=f'Date Sourced: {str(df_list[i].index[1])}', 
                                                          rot=50, xlabel='Date Published', 
                                                          ylabel='Number of Articles', ax=ax_arr[i]);
        ax_arr[i].legend(['Articles'])
        
#     plt.subplot_tool();



"""
Data Cleaning funcitons
""" 

def cleaned_articles(archive):
    cleaned = []
    
    # loop through every article and append to empty list 
    for article in archive:
        uri = article['uri']
        date_published = article['pub_date'][:10]
        headline = article['headline']['main'].lower()
        keywords = ''.join(x['value'].lower() for x in article['keywords'])
        snippet = article['snippet']
        word_count = article['word_count']
        cleaned.append([uri, date_published, headline, keywords, snippet, word_count])
        
    return cleaned


# function to extract only needed information and make strings lowercase 
def cleaned_shared_articles(most_shared_articles):
    cleaned_articles = []
    
    # loop through every article and append to empty list 
    for article in most_shared_articles:
        date = pd.Timestamp("today").strftime("%m/%d/%Y")
        idx = pd.to_datetime(date)
        uri = article['uri']
        date_published = pd.to_datetime(article['published_date'])
        cleaned_articles.append([idx, uri, date_published])
        
    return cleaned_articles


# function to load and combine all most shared articles 
def load_most_shared(dir_path):

    directory = os.fsencode(dir_path)
    
    files = []
    most_shared_df = pd.DataFrame()
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.csv'):
            files.append(filename)
            
    #read them into pandas
    df_list = [pd.read_csv(dir_path+'/'+file) for file in files]
    
    #concatenate them together
    df = pd.concat(df_list)
    
    df.date_published = df.date_published.apply(lambda x: pd.to_datetime(x).date())
    df.date_sourced = df.date_sourced.apply(lambda x: pd.to_datetime(x).date())
    df.set_index('date_sourced', inplace=True)
    
    return df    

    
    
"""
Sampling funciton
""" 

def smote_data(X_train, y_train, sampling_strategy, random_state):
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_sample(X_train, y_train) 
    return X_train_resampled, y_train_resampled 




"""
NLP functions
""" 
# This function gets the correct Part of Speech so the Lemmatizer in the next function can be more accurate.
def get_wordnet_pos(treebank_tag):
    '''
    Translate nltk POS to wordnet tags
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# param headline: a single headline
# return: a headline string with words which have been lemmatized, parsed for 
# stopwords and stripped of punctuation and numbers.
def text_prep(text, sw):
    
    sw = stopwords.words('english')
    regex_token = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
    text = regex_token.tokenize(text)
    text = [word for word in text]
    text = [word for word in text if word not in sw]
    text = pos_tag(text)
    text = [(word[0], get_wordnet_pos(word[1])) for word in text]
    lemmatizer = WordNetLemmatizer() 
    text = [lemmatizer.lemmatize(word[0], word[1]) for word in text]
    return ' '.join(text)


def vectorize_feature(vectorizer, X_train, X_test):
    sw = stopwords.words('english')
    vectorizer = vectorizer(stop_words=sw)
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    return vectorizer, X_train_vec, X_test_vec


def tokenize_vector(vectorizer, X_train, X_test):
    sw = stopwords.words('english')
    
    vectorizer = vectorizer(stop_words=sw)
    
    X_train_tokenized = [text_prep(text, sw) for text in X_train]
    X_train_token_vec = vectorizer.fit_transform(X_train_tokenized)
    X_test_token_vec = vectorizer.transform(X_test)
    
    return vectorizer, X_train_token_vec, X_test_token_vec


def plot_top_words(vectorizer, X_train, title_suffix):
    
    df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names())
    limit = 5

    plt.figure(figsize=(5,5))
    plt.barh(df.sum().sort_values(ascending=False)[:limit].index, 
             df.sum().sort_values(ascending=False)[:limit])
    plt.title(f'Top {str(limit)} Words: {title_suffix}')
    plt.xlabel('Word Count')
    plt.ylabel('Words');


    