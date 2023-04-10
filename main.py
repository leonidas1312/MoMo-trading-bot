import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import streamlit_toggle as tog
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np

# Define the list of cryptocurrencies and stocks to display
crypto_list = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'XRP-USD', 'DOGE-USD', 'BNB-USD', 'SOL1-USD', 'LUNA1-USD', 'LINK-USD',
               'XLM-USD', 'MATIC-USD', 'DOT1-USD', 'AVAX-USD', 'ATOM1-USD', 'ALGO-USD']
stock_list = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'FB', 'BRK-A', 'NVDA', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD',
              'DIS']

# Set default start date to 6th of April 2023
default_start_date = datetime(2023, 4, 6)


# Define neural network architecture
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x


def display_prices_ma(symbol, start_date, end_date, interval, short_window, long_window):
    try:
        # Get data from Yahoo Finance API
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        # Calculate moving averages
        data["SMA"] = data["Close"].rolling(short_window).mean()
        data["LMA"] = data["Close"].rolling(long_window).mean()
        data['EMA'] = data['Close'].ewm(span=10, adjust=False).mean()  # e moving average

        # print(data)
        # Check for NaN values

        # Drop rows with missing data
        data.dropna(inplace=True)
        # Create plot using Plotly Express
        fig = px.line(data, x=data.index, y=["Close", "SMA", "LMA", "EMA"],
                      labels={"value": "Price", "variable": "Metric"},
                      title=f"{symbol} Price and Moving Averages")
        fig.update_layout(legend=dict(x=1, y=1), margin=dict(l=20, r=20, t=60, b=20), height=500)
        # Show plot in Streamlit app
        st.plotly_chart(fig)
        return data
    except ValueError as e:
        if interval == "1m":
            st.warning("For 1 minute interval only 7 days worth of data can be used")
        elif interval == "5m":
            st.warning("For 5 minute interval only 60 days worth of data can be used")
        elif interval == "15m":
            st.warning("For 15 minute interval only 60 days worth of data can be used")
        elif interval == "30m":
            st.warning("For 30 minute interval only 60 days worth of data can be used")
        elif interval == "60m":
            st.warning("For 60 minute interval only 730 days worth of data can be used")


# Calculate the moving averages for a given symbol and time period
def calculate_moving_averages(symbol, start_date, end_date, short_window, long_window):
    df = yf.download(symbol, start=start_date, end=end_date)
    df['SMA_short'] = df['Close'].rolling(short_window).mean()
    df['SMA_long'] = df['Close'].rolling(long_window).mean()
    return df[['Close', 'SMA_short', 'SMA_long']]


# Define the Streamlit app
st.title('MoMo trading bot')
with st.container():
    st.markdown("""
                - Live backtesting and simulation of a trading strategy
                - Trade stocks or cryptocurrencies
    
    
                """)
    st.write("---")

# sidebar
selected_crypto = st.sidebar.multiselect('Select cryptocurrencies:', crypto_list, default=['BTC-USD'])
selected_stock = st.sidebar.multiselect('Select stocks:', stock_list, default=['AAPL'])
start_date = st.sidebar.date_input('Start date:', value=default_start_date)
end_date = st.sidebar.date_input('End date:', value=None)
interval = st.sidebar.selectbox('Select time interval:', [1, 5, 15, 30, 60], format_func=lambda x: f'{x} minutes')
columns = st.sidebar.multiselect('Select columns:', ['Open', 'High', 'Low', 'Close'], default=['Close'])

# main window
short_window = st.number_input("Input number of timesteps for the short moving average window", min_value=1,
                               max_value=365, value=50)
long_window = st.number_input("Input number of timesteps for the long moving average window", min_value=1,
                              max_value=365, value=200)

# Display prices for selected cryptocurrencies and stocks side by side
# col1, col2 = st.columns(2)

tt = tog.st_toggle_switch(label="Display timeseries data",
                          key="Key1",
                          default_value=False,
                          label_after=False,
                          inactive_color='#D3D3D3',
                          active_color="#FEA501",
                          track_color="#FEA501",
                          )

# for crypto in selected_crypto:
#   fig_crypto = display_prices(crypto, start_date, end_date, f'{interval}m', columns)
#   st.plotly_chart(fig_crypto, use_container_width=True)
# for stock in selected_stock:
#    fig_stock = display_prices(stock, start_date, end_date, f'{interval}m', columns)
#    st.plotly_chart(fig_stock, use_container_width=True)

if tt:
    for crypto in selected_crypto:
        data_c = display_prices_ma(crypto, start_date, end_date, f'{interval}m', short_window, long_window)

    for stock in selected_stock:
        data_s = display_prices_ma(stock, start_date, end_date, f'{interval}m', short_window, long_window)

if st.sidebar.button("Run"):

    # Define hyperparameters
    n_days = 24
    input_size = n_days * 1 + 3  # number of input features
    hidden_size = 64  # number of hidden units
    output_size = 1  # number of output units
    learning_rate = 0.01
    num_epochs = 1000

    # Preprocess data
    time_interval = '60min'
    for i in range(1, n_days + 1):
        data_c[f'Previous_Close_{i}'] = data_c['Close'].shift(
            int(i * pd.Timedelta(time_interval).total_seconds() / 3600))

    data_c.dropna(inplace=True)

    # Split data into train and test sets
    train_size = int(0.7 * len(data_c))
    train_data = data_c.iloc[:train_size]
    test_data = data_c.iloc[train_size:]

    # Normalize data
    scaler = torch.nn.BatchNorm1d(input_size)
    X_train = scaler(torch.tensor(train_data.iloc[:, -input_size:].values).float())
    y_train = torch.tensor(train_data['Close'].values).float().view(-1, 1)
    X_test = scaler(torch.tensor(test_data.iloc[:, -input_size:].values).float())
    y_test = torch.tensor(test_data['Close'].values).float().view(-1, 1)

    # Initialize neural network
    net = Net(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Forward pass
        outputs = net(X_train)
        loss = criterion(outputs, y_train)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # Test NN
    net.eval()
    with torch.no_grad():
        predicted = net(X_test)
    test_loss = criterion(predicted, y_test)

    print('Test Loss: {:.4f}'.format(test_loss.item()))
    print(predicted)
    print(len(predicted))
    predicted_next_hour = predicted[-1].item()
    print(predicted_next_hour)
    fig = px.line(test_data, y='Close', labels={'Close': 'Price'})
    fig.add_scatter(x=data_c.index[-1],y=[predicted_next_hour], name='Predicted')
    st.plotly_chart(fig)

    # Predict next 24 hours
    last_date = data_c.index[-1]
    input_24h = []
    for i in range(1, 25):
        input_24h.append(data_c.loc[last_date + pd.Timedelta(hours=i), ['Open', 'High', 'Low', 'Close', 'Volume']])
    input_24h = pd.concat(input_24h, axis=1).T
    for i in range(1, n_days + 1):
        input_24h[f'Previous_Close_{i}'] = input_24h['Close'].shift(i)

    input_24h.dropna(inplace=True)
    print(input_24h)
    X_24h = scaler(
        torch.tensor(input_24h.iloc[:, -input_size:].values).float())
    with torch.no_grad():
        predicted_24h = net(X_24h)

    print('Predicted prices for next 24 hours:')
    print(predicted_24h.numpy().flatten())

