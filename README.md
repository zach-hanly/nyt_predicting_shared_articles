# Predicting The New York Time's Most Shared Articles

**Author**: Zach Hanly

## Overview

This project takes article information from the New York Times and uses NLP, Random Forests and Logistic Regression to predict the daily Top 20 most shared articles on Facebook. 

## Business and Data Understanding

In order to show how social media popularity for text media can be predicted, metadata on articles from the New York Time's API website was gathered.

Two APIs were used. First, the __[NYT Archive API](https://developer.nytimes.com/docs/archive-product/1/overview)__,  which returns metadata on every article for a given range of months and years. Second, the __[NYT Most Popular API](https://developer.nytimes.com/docs/most-popular-product/1/overview)__, which returns metadata on the most popular articles on NYTimes.com based on views on the NYT site, emails, or Facebook shares. For this project Facebook shares was chosen and calls were made to the API once per day to gather that day's top 20 list. Articles in the archive that were listed on a top 20 list were then labeled as a popular article for a modeling target. 

## Modeling

Two identical looking pipelines were used. One using a TfidfVectorizer which converted text features to floating point values and the other using a CountVectorizer that converted text to binary values. Each feature then went into its own model, where the TfidfVectorizer pipeline used a Random Forest to turn article features into a probability that the article would be on the top 20 most shared on Facebook list, while the CountVecotized pipeline used a Random Forrest to turn them into a binary class label. Both pipelines end at a Logistic Regression model that outputs the probability that the article with all its converted features will be a top 20 article. This final model is where the word count feature is joined in from its Logistic Regression model, which turned the numeric values into either a probability or a class label for the respective pipelines. 

#### Diagram
![model diagram](images/model_diagram.png)



After training and testing the pipelines with a train-test split, the pipelines attempted to predict several day's Top 20 list. 
#### Results 
![model results](images/model_results.png)

## Conclusion

The pipeline using the TfidfVectorizer modeled to probabilities outperformed the pipeline using the CountVectorizer modeled to class labels. 


## For More Information

Please review the full analysis in my [Jupyter Notebook](./main_notebook.ipynb) or my [presentation](./presentation.pdf).

## Repository Structure

```
├── data                      <- Sourced externally from APIs and generated from modeling 
├── images                    <- Sourced externally and generated from code
├── notebooks                 <- building scripts for API calls and functions.py
├── .gitignore                <- files github should ignore 
├── README.md                 <- The top-level README for reviewers of this project
├── functions.py              <- created functions for project  
├── main_notebook.ipynb       <- Narrative documentation of modeling
├── request_archive.py        <- collects arichive of articles
├── request_most_shared.py    <- collects daily top 20 most shared article son facebook
├── presentation.pdf          <- PDF version of project 
```# nyt_predicting_sharable_article
