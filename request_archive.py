""" This script will make a call to the NYT API for the all articles in a given month(s).
"""

# import required packages
from functions import *   
import requests 
import json
import pandas as pd
  

# make a list of months than pass to API call function 
"""switch to article_dates_deploy when ready to stop training.
"""
article_dates_train = [(2021, 12), (2022, 1)]
article_dates_deploy = [(2022, 1)]
articles = get_articles(article_dates_deploy)

# pass list of articles through cleaning function 
cleaned_articles = cleaned_articles(articles)

# put articles in dataframe and drop duplictes, if any 
df_articles = pd.DataFrame(cleaned_articles)


# rename columns to strings 
df_articles.columns = ['uri', 'date_published', 'headline', 'keywords', 'snippet', 'word_count']

# convert date_published to datetime dtype
df_articles.date_published = df_articles.date_published.apply(lambda x: pd.to_datetime(x).date())

    # save training data to csv file
    # df_articles.to_csv('data/archive_train.csv', index=False)

    
"""uncomment and unindent next two lines to only accept articles with dates after training phase
"""
last_training_day = pd.to_datetime('2022/01/15').date()
df_articles = df_articles[df_articles.date_published > last_training_day]    
    

""" uncoment and unindent next line to save delpoyment archive data to a seperate file. 
    comment line above to prevent delpoyment data from leaking into training file.
"""
df_articles.to_csv('data/archive_deploy.csv', index=False)