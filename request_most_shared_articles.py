""" This script will make a call to the NYT API for the top 20 articles shared on facebook for one day.
    This must be ran once per day to get the newest top 20 list. 
"""

# import required packages
from functions import *   
import requests 
import json
import pandas as pd


# get current day to label file with the day it was accessed 
date_sourced = pd.Timestamp("today").strftime("%m/%d/%Y")
date_sourced = date_sourced.replace('/','_')

""" In order to train and deploy the model under time constraints, the indented and commented 
    code below served to obtain a top 20 most-shared articles list for the previous 30 days. 
    This reflects how the project was started, but if repeated, only daily top 20 lists should be used.  
"""
    # most_shared_before = get_most_shared_articles(30)

    # cleaned_most_shared_before = cleaned_shared_articles(most_shared_before)

    # df_shared_before = pd.DataFrame(cleaned_most_shared_before)

    # df_shared_before.columns = ['date_sourced', 'uri', 'date_published']

    # df_shared_before.to_csv(f'data/most_popular_train/most_shared_before{date_sourced}.csv', index=False)

    
""" Daily top 20 list starts below 
"""    
# make API call with function to get most shared articles from previous day 
most_shared_articles = get_most_shared_articles(1)

# pass list of articles through cleaning function 
cleaned_most_shared = cleaned_shared_articles(most_shared_articles)

# put articles in dataframe
df_shared = pd.DataFrame(cleaned_most_shared)

# rename columns to strings 
df_shared.columns = ['date_sourced', 'uri', 'date_published']

# save file to folder in data directory 
# df_shared.to_csv(f'data/most_popular_train/most_shared_{date_sourced}.csv', index=False) 


""" uncoment and unindent next line to divert top 20 lists for deployment into a different folder.
    comment line above to prevent delpoyment data from leaking into training folder.
"""
df_shared.to_csv(f'data/most_popular_deploy/most_shared_{date_sourced}.csv', index=False) 