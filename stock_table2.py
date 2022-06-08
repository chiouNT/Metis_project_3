"""

Engineering project of Ni-Ting Chiou
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import yfinance as yf
from datetime import date

st.title("Stock price change predictions")

st.write('''
### Using the engineered close price features for the prediction
''')

st.sidebar.markdown("Select from the menus below:")
dataset_name = st.sidebar.selectbox("Select Dataset", 
                                   ("sp500", "Seaboard", "Amazon", "Booking_Holdings", "Google"))
                                
price_name = st.sidebar.selectbox("Select Predictor Name", 
                                      ("Close",
                                      "Ratio_2D", 
                                      "Ratio_5D", 
                                      "Ratio_60D",
                                      "Ratio_250D",
                                      "Ratio_1000D",
                                      "Trend_2D",
                                      "Trend_5D",
                                      "Trend_60D",
                                      "Trend_250D",
                                      "Trend_1000D"))

url = "https://raw.githubusercontent.com/chiouNT/Engineering/main/stock_data_github.csv"
data_stock = pd.read_csv(url)
data_stock["Date"] = pd.to_datetime(data_stock["Date"])
data_stock = data_stock.set_index("Date")

def load_clean_data(ticker_name):

    dataset =  data_stock[data_stock["name"] == ticker_name].copy()
    
    dataset['Tomorrow'] = dataset['Close'].shift(-1)
    dataset['Target'] = (dataset['Tomorrow'] > dataset['Close']).astype(int)    
    del dataset['Tomorrow']
    del dataset['id']
    
    horizons =[2, 5, 60, 250, 1000]
    new_predictors = []
    
    for horizon in horizons:
        rolling_averages = dataset.rolling(horizon).mean()
        ratio_column = f"Ratio_{horizon}D"
        
        dataset[ratio_column] = round(dataset['Close']/rolling_averages['Close'], 1)
        
        trend_column= f'Trend_{horizon}D'
        dataset[trend_column] = round(dataset.shift(1).rolling(horizon).sum()["Target"],1)
        
        new_predictors += [ratio_column, trend_column]
    dataset = dataset.dropna() 
    return dataset


def load_dataset(dataset_name):
    if dataset_name.lower() == "sp500":
        data = load_clean_data("^GSPC")

    elif dataset_name.lower() == "seaboard":
        data = load_clean_data("SEB")
    elif dataset_name.lower() == "amazon":
        data = load_clean_data("AMZN")
    elif dataset_name.lower() == "booking_holdings":
        data = load_clean_data("BKNG")
    else:
        data = load_clean_data("GOOG")

    return data


data = load_dataset(dataset_name)
st.write(f"Shape of the dataset: {data.shape}")




# Plotting
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(data["Close"])
plt.xlabel("Date")
plt.ylabel("price")
plt.title("All data")


plt.subplot(2,1,2)
plt.plot(data.iloc[-2750:][{price_name}])
plt.xlabel("Date")
plt.ylabel("price")
plt.title("Data in the recent 10 years (used for predictions)")
fig.tight_layout()
st.pyplot(fig)

# Predictions
new_predictors = list(data.columns[7:])

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


model  = RandomForestClassifier(n_estimators=600, min_samples_split=50, random_state =1)
train = data.iloc[-2750:-250].copy()
test = data.iloc[-250:].copy()
model.fit(train[new_predictors], train["Target"])
predictions = predict(train, test, new_predictors, model)
precision_scores = precision_score(predictions["Target"], predictions.Predictions)

# Displaying the accuracy and the model details

st.write(f"precision_scores: {precision_scores:.2%}")
st.write(
f'Probability that the price will increase in the day next to {data[-1:].reset_index().iloc[0, 0,].strftime("%m/%d/%Y")} is {model.predict_proba(data[new_predictors][-1:])[:, 1][0]:.2%}'
)

