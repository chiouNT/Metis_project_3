# Engineering project: Stock market prediction web application

## Abstract

The goal of the project is to build a web application for characterizing the stock close price and predicting whether the stock close price will go or down tomorrow. The engineered close price features and RandomForestClassifier were used for the stock price change prediction. To build the web application, the stock data was ingested from Yahoo Finance API, stored at SQLite, processed at AWS and deployed at streamlit.io. The web app allows the users to choose the stock database for analyze its trend of the close price, precision score and probability of the price going up in the next day.  

## Design

The target of the prediction is whether the stock prices will go up or down tomorrow. To predict the stock price changes, the close price was engineered to give 10 engineered ratio and trend features. For 5 ratio features, they are the ratio between today’s close price and the averaged close price in the past 2, 5, 60, 250 and 1000 days. For 5 trend features, they are the number of days that have increased close price during the past 2, 5, 60, 250 and 1000 days.

## Data Pipeline

The data is ingested from Yahoo’s Finance API and then stored at SQLite database at AWS. The data at database is then further processed by jupyter notebook at AWS and the resulting stock dataframe is pushed to Github. There are 4 steps during the deployment: (1) the dataframe is read from Github, (2) the close prices are engineered to give 10 features, (3) Data plotting of the close price and the engineered features and (4) Modeling and prediction of the close price change in the next day.  

## Algorithms

*Feature Engineering*
* Rolling window calculation is used for computing the ratio and trend of close price for each day.

*Models* <br> 
* RandomForestClassifier

*Unit test* <br>
* Test the function for loading the stock database from API

## Tools
*	SQL, AWS cloud, and streamlit
*	Pandas, Numby, Matplotlib and Seaborn: Data clean, exploration and visualization
*	Sci-learn learn: predictive data analysis


## Communication
* The following is the web application of the stock market prediction. 
[Please use Chrome to open it](https://share.streamlit.io/chiount/engineering/main)
