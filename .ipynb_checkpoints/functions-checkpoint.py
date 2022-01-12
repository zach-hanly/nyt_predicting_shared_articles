 # import needed libraries
import os
import requests 
import json
import pandas as pd
import datetime as dt  


"""
file functions
""" 

#function to access private API key
def get_api_key(path):
    with open(path) as f:
        return json.load(f)
    

    
"""
API request functions
""" 

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
Data Cleaning funciton
""" 

def cleaned_articles(archive):
    cleaned = []
    
    # loop through every article and append to empty list 
    for article in archive:
        uri = article['uri']
        date_published = article['pub_date']
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

# function to laod and combine all most shared articles 
def load_most_shared(dir_path):
#     'data/most_popular'
    directory = os.fsencode(dir_path)
    
    files = []
    most_shared_df = pd.DataFrame()
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.csv'):
            files.append(filename)
            
    #read them into pandas
    df_list = [pd.read_csv('data/most_popular/'+file) for file in files]
    
    #concatenate them together
    most_shared_df = pd.concat(df_list)
    most_shared_df.set_index('date_sourced', inplace=True)
    
    return most_shared_df
    


"""
Sampling funcitons 
""" 

def smote_data(X_train, y_train, sampling_strategy, random_state):
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_sample(X_train, y_train) 
    return X_train_resampled, y_train_resampled 


"""
NLP functions
""" 

# This function the correct Part of Speech so the Lemmatizer in the next function can be more accurate.
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
def text_prep(text, stop_words=sw):
    
    regex_token = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
    text = regex_token.tokenize(text)
    text = [word for word in text]
    text = [word for word in text if word not in sw]
    text = pos_tag(text)
    text = [(word[0], get_wordnet_pos(word[1])) for word in text]
    lemmatizer = WordNetLemmatizer() 
    text = [lemmatizer.lemmatize(word[0], word[1]) for word in text]
    return ' '.join(text)