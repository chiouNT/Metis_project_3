Minimum Viable Product (Ni-Ting Chiou)

##  sp500 index prediction 

* **Data ingestion:** Yahoo finance API

* **Processing:** 
  * Target: The close price for the next day is higher than today.
  * Features: 
    * The ratio of the close price between a day and the average of its previous 2, 5, 60, 250, or 1000 days.
    * The sum of days that have the hihger close price for the next day during the past 2, 5, 60, 250, or 1000 days.
  * Model: RandomForestClassifier

* **Deployment:** Streamlit


#  A web application for predicting whether the close index of sp500 will go up tomorrow


[Link for seeing the web app (please use Chrome to open it)](https://share.streamlit.io/chiount/stock_prediction/main/stock_streamlit_app.py)


