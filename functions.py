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
import cv2
from wordcloud import WordCloud 

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
    
    Deploy Function 
    
    
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


def top_20_delta_perc(df_list):
    
    date_list = []
    for df in df_list:
        day_sourced = df.index[0]
        value_count = df.date_published.value_counts()
        time_delta = [day_sourced - date for date in value_count.index]
        date_dict = {k:v for k,v in list(zip(time_delta, value_count.values))}
        date_series = pd.Series(date_dict, name=str(day_sourced))
        date_list.append(date_series)
        
    top20_delta_df = pd.DataFrame(date_list).T
    num_days = len(top20_delta_df.columns)
    top20_delta_df['percentage'] = 0
    top20_delta_df.fillna(0, inplace=True)
    top20_delta_df['percentage'] = round((top20_delta_df.sum(axis=1))/(num_days*20), 4)
    delta = top20_delta_df.percentage
    
    X = [str(x)+' days' for x in list(delta.index.days)]
    plt.bar(X, delta.values)
    plt.title('Time Delta Distribution', size=20, pad=10)
    plt.xlabel('Days Published Before Day on Top 20', size=13, labelpad=10)
    plt.ylabel('Percentage of Total Articles', size=13, labelpad=10)
    plt.xticks(size=10)
    plt.yticks([0, .10, .20, .30, .40, .50], labels=['0%', '10%', '20%', '30%', '40%', '50%'], size=10)
    plt.legend(['Articles'], prop={'size': 10});
    
  
    return delta.round(2).apply(lambda x: str(int(x*100))+'%').to_frame()



def plot_word_cloud(text_column, mask_path):
    
    image = cv2.imread(mask_path, 1)
    text = ' '.join([x for x in text_column])
    
    wordcloud = WordCloud(background_color=None, mask=image, mode = "RGBA", 
                          max_words=100, height=1000, random_state=0).generate(text)
    plt.figure(figsize=(10,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")



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


    
"""
Deploy functions 
"""

# function to load most shared articles for eda
def load_predictions(dir_path):
    
    directory = os.fsencode(dir_path)
    
    files = []
    pred_df = pd.DataFrame()
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.csv'):
            files.append(filename)
            
    #read them into pandas
    df_list = [pd.read_csv(dir_path+'/'+file) for file in files]
    plot_df = pd.concat(df_list)
    plot_df.set_index(plot_df.columns[0], inplace=True)
    plot_df.index.name = 'date'
    
    return plot_df


def plot_predictions(df):
    
    times = pd.date_range(df.index[0], periods=len(df.index))

    proba_tf_pipe = df.proba_tf_pipe
    class_cv_pipe = df.class_cv_pipe
    df = pd.DataFrame({'Tfidf Pipeline': df.proba_tf_pipe, 'CountVectorizer Pipeline': df.class_cv_pipe})

    ax = df.plot.bar(color=['SkyBlue','IndianRed'], title='Too 20 Articles Predictions')
    plt.xticks(rotation=45)

    ax.set_xlabel('Prediction Day')
    ax.set_ylabel('Percentage Correct')
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5], 
                   ['0%', '10%', '20%', '30%', '40%', '50%'])
              

        
def score_top20_pred(top_20_from_proba, top_20_from_class, df_top_20):
    
    date = df_top_20.index[0]
    
    proba_tf_pipe = [1 if x in df_top_20.uri.values else 0 for x in top_20_from_proba.uri.values]
    proba_tf_pipe_perc = sum(proba_tf_pipe)/len(proba_tf_pipe)
    class_cv_pipe = [1 if x in df_top_20.uri.values else 0 for x in top_20_from_class.uri.values]
    class_cv_pipe_perc = sum(class_cv_pipe)/len(class_cv_pipe)
    
    pred_series = pd.Series({'proba_tf_pipe': proba_tf_pipe_perc, 'class_cv_pipe': class_cv_pipe_perc}, 
                            name=str(date))
    
    feedback = pd.DataFrame()
    
    return pred_series



# function to load most shared articles for eda
def load_predictions(dir_path):
    
    directory = os.fsencode(dir_path)
    
    files = []
    pred_df = pd.DataFrame()
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.csv'):
            files.append(filename)
            
    #read them into pandas
    df_list = [pd.read_csv(dir_path+'/'+file) for file in files]
    plot_df = pd.concat(df_list)
    plot_df.set_index(plot_df.columns[0], inplace=True)
    plot_df.index.name = 'date'
    
    return plot_df
              