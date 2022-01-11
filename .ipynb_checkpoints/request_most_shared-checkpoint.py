""" This script will make a call to the NYT API for the articles that were 
    the most shared on facebook.
    Code to save datframes to csv should be uncommented to when ready to run. 
    data goe sinto seperete 'data' folder  
"""

""" In order to initialize the model with data for article son the most shared list 
    prior to collecting data, uncomment and unindent the indented code. Otherwise the only 
    articles able to be labeled as a most shared article will be those accessed 
    the day the project is started and onward
"""

# import needed libraries
from functions import *   
import requests 
import json
import pandas as pd


    # most_shared_before = get_most_shared_articles(30)

    # cleaned_most_shared_before = cleaned_shared_articles(most_shared_before)

    # df_shared_before = pd.DataFrame(cleaned_most_shared_before)

    # df_shared_before.columns = ['date_sourced', 'uri', 'date_published']

# access current day to label data with the day it was accessed 
date_sourced = pd.Timestamp("today").strftime("%m/%d/%Y")
date_sourced = date_sourced.replace('/','_')

# df_shared_before.to_csv(f'data/most_popular/most_shared_before{date_sourced}.csv', index=False)

# make API call with function to get most shared articles from previous day 
most_shared_articles = get_most_shared_articles(1)

# pass list of articles through cleaning function 
cleaned_most_shared = cleaned_shared_articles(most_shared_articles)

# put articles in dataframe
df_shared = pd.DataFrame(cleaned_most_shared)

# rename columns to strings 
df_shared.columns = ['date_sourced', 'uri', 'date_published']

df_shared.to_csv(f'data/most_popular/most_shared_{date_sourced}.csv', index=False) 