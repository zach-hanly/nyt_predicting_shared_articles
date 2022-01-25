# Predicting New York Times Daily Top 20 Most Shared Articles on Facebook 

**Author**: Zach Hanly

## Overview

This project takes article inofmration from the New York times and uses NLP, Random Forest's and logistic regression to predict the daily top 20 most shared articles on Facebook. 

## Business and Data Understanding

In order to show how social media popularity for textual media can be modeled, article metadata was gathered from the New York Time's API website.

Two APIs were used. First, the __[NYT Archive API](https://developer.nytimes.com/docs/archive-product/1/overview)__,  which returns every article for a given range of months and years. Second, the __[NYT Most Popular API](https://developer.nytimes.com/docs/most-popular-product/1/overview)__, which returns the most popular articles on NYTimes.com based on emails, Facebook shares, or views on the NYT site. For this project Facebook shares were chosen and calls were made to the API once per day to gather that day's top 20 list. Articles in the archive that were listed on a top 20 list were then labeled as a popular article for a modeling target. 

## Modeling

![model results](images/model_results.png)

## Conclusion



## For More Information

Please review the full analysis in my [Jupyter Notebook](./main_notebook.ipynb) or my [presentation](./presentation.pdf).

## Repository Structure

```
├── data                      <- Sourced externally
├── images                    <- Generated from code
├── .gitignore                <- filed github should ignore  
├── notebook.ipynb            <- Narrative documentation of modeling
├── README.md                 <- The top-level README for reviewers of this project
├── presentation.pdf          <- PDF version of project 
```# nyt_predicting_sharable_article
