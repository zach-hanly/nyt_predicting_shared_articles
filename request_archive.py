""" This script will make a call to the NYT API for the all articles in a given month
"""

# import needed libraries
from functions import *   
import requests 
import json
import pandas as pd
  

# make a list of months than pass to API call function 
article_dates = [(2021, 12), (2022, 1)]
articles = get_articles(article_dates)

# pass list of articles through cleaning function 
cleaned_articles = cleaned_articles(articles)

# put articles in dataframe and drop duplictes, if any 
df_articles = pd.DataFrame(cleaned_articles)
df_articles.drop_duplicates(0, inplace=True)

# rename columns to strings 
df_articles.columns = ['idx', 'headline', 'keywords', 'author', 'section', 'word_count']

df_articles.to_csv('data/article_archive.csv')