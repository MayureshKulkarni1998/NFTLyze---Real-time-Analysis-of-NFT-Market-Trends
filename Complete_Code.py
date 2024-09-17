# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import openpyxl as pxl
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image, ImageTk
from statsmodels.tsa.arima.model import ARIMA

#import data from source excel file
df_raw = pd.read_csv('NFTlyze Dataset.csv', parse_dates=['Date'])             # Read any input data through pd.read using Pandas

# Handle Missing Values
df = df_raw.dropna()                                                          # Handling missing values by dropping rows with NAN values
df.set_index('Date', inplace=True)                                            # Created dataframe with first column as Index column
df1 = df_raw.dropna()                                                         # Handling missing values by dropping rows with NAN values 

# EXPLORATORY DATA ANALYSIS

# 1. Visualization - Cumulative Metric Plots over time
fig, axs = plt.subplots(2, 2, figsize=(20, 16))
# Visualization - Cumulative Sales USD over Time
axs[0, 0].plot(df.index, df['Sales_USD_cumsum'], label='Sales_USD_cumsum', color='blue')
axs[0, 0].set_title('Cumulative Sales USD over Time')
axs[0, 0].set_xlabel('Date')
axs[0, 0].set_ylabel('Sales_USD_cumsum')
# Visualization - Cumulative Average USD per Sale over Time
axs[0, 1].plot(df.index, df['Sales_USD_cumsum'], label='Sales_USD_cumsum', color='green')
axs[0, 1].set_title('Cumulative Average USD per Sale over Time')
axs[0, 1].set_xlabel('Date')
axs[0, 1].set_ylabel('AverageUSD_cum')
# Visualization - Cumulative Number of Sales over Time
axs[1, 0].plot(df.index, df['Number_of_Sales_cumsum'], label='Number_of_Sales_cumsum', color='red')
axs[1, 0].set_title('Cumulative Number of Sales over Timem')
axs[1, 0].set_xlabel('Date')
axs[1, 0].set_ylabel('Number_of_Sales_cumsum')
# Visualization - Cumulative Active Market Wallets over Time
axs[1, 1].plot(df.index, df['Active_Market_Wallets_cumsum'], label='Active_Market_Wallets_cumsum', color='purple')
axs[1, 1].set_title('Cumulative Active Market Wallets over Time')
axs[1, 1].set_xlabel('Date')
axs[1, 1].set_ylabel('Active_Market_Wallets_cumsum')

plt.tight_layout()
plt.savefig('Cumulative Metrics.png')
plt.show()

# 2. Visualization - Actual Metric Plots over time
fig, axs = plt.subplots(2, 2, figsize=(20, 16))
# Visualization - Actual Sales USD over Time
axs[0, 0].plot(df.index, df['Sales_USD'], label='Sales_USD', color='blue')
axs[0, 0].set_title('Actual Sales USD over Time')
axs[0, 0].set_xlabel('Date')
axs[0, 0].set_ylabel('Sales_USD')
# Visualization - Actual Number of Sales over Time
axs[0, 1].plot(df.index, df['Number_of_Sales'], label='Number_of_Sales', color='green')
axs[0, 1].set_title('Actual Number of Sales over Time')
axs[0, 1].set_xlabel('Date')
axs[0, 1].set_ylabel('Number_of_Sales')
# Visualization - Actual Active Market Wallets over Time
axs[1, 0].plot(df.index, df['Active_Market_Wallets'], label='Active_Market_Wallets', color='red')
axs[1, 0].set_title('Actual Number of Sales over Timem')
axs[1, 0].set_xlabel('Date')
axs[1, 0].set_ylabel('Active_Market_Wallets')
# Visualization - Actual Primary Sales over Time
axs[1, 1].plot(df.index, df['Primary_Sales'], label='Primary_Sales', color='purple')
axs[1, 1].set_title('Actual Primary Sales over Time')
axs[1, 1].set_xlabel('Date')
axs[1, 1].set_ylabel('Primary_Sales')

plt.tight_layout()
plt.savefig('Actual Metrics.png')
plt.show()

# 3. Visualization - Correlation matrix heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, cmap='coolwarm', fignum=1)
plt.title('Correlation Matrix', pad=20)
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

plt.savefig('Correlation Matrix - Heat Map.png')
plt.show()

# 4. Visualization - Monthly Sales Data

fig, axs = plt.subplots(2, 2, figsize=(20, 16))
monthly_sales_usd_cum = df['Sales_USD_cumsum'].resample('M').sum()
axs[0, 0].plot(monthly_sales_usd_cum.index, monthly_sales_usd_cum, label='Sales_USD_cumsum', color='blue')
axs[0, 0].set_title('Cumulative Sales USD - Monthly')
axs[0, 0].set_xlabel('Date')
axs[0, 0].set_ylabel('Sales_USD_cumsum')
monthly_sales_cum = df['Number_of_Sales_cumsum'].resample('M').sum()
axs[0, 1].plot(monthly_sales_cum.index, monthly_sales_cum, label='Number_of_Sales_cumsum', color='green')
axs[0, 1].set_title('Cumulative Number of Sales - Monthly')
axs[0, 1].set_xlabel('Date')
axs[0, 1].set_ylabel('Number_of_Sales_cumsum')
monthly_sales_usd = df['Sales_USD'].resample('M').sum()
axs[1, 0].plot(monthly_sales_usd.index, monthly_sales_usd, label='Sales_USD', color='blue')
axs[1, 0].set_title('Sales USD - Monthly')
axs[1, 0].set_xlabel('Date')
axs[1, 0].set_ylabel('Sales_USD')
monthly_sales = df['Number_of_Sales'].resample('M').sum()
axs[1, 1].plot(monthly_sales.index, monthly_sales, label='Number_of_Sales', color='green')
axs[1, 1].set_title('Number of Sales - Monthly')
axs[1, 1].set_xlabel('Date')
axs[1, 1].set_ylabel('Number_of_Sales')

plt.tight_layout()
plt.savefig('Monthly Data - Sales USD and Number of Sales.png')
plt.show()

# 5. Visualization - Yearly Sales Data

fig, axs = plt.subplots(2, 2, figsize=(20, 16))
yearly_sales_usd_cum = df['Sales_USD_cumsum'].resample('Y').sum()
axs[0, 0].plot(yearly_sales_usd_cum.index, yearly_sales_usd_cum, label='Sales_USD_cumsum', color='blue')
axs[0, 0].set_title('Cumulative Sales USD - Yearly')
axs[0, 0].set_xlabel('Date')
axs[0, 0].set_ylabel('Sales_USD_cumsum')
yearly_sales_cum = df['Number_of_Sales_cumsum'].resample('Y').sum()
axs[0, 1].plot(yearly_sales_cum.index, yearly_sales_cum, label='Number_of_Sales_cumsum', color='green')
axs[0, 1].set_title('Cumulative Number of Sales - Yearly')
axs[0, 1].set_xlabel('Date')
axs[0, 1].set_ylabel('Number_of_Sales_cumsum')
yearly_sales_usd = df['Sales_USD'].resample('Y').sum()
axs[1, 0].plot(yearly_sales_usd.index, yearly_sales_usd, label='Sales_USD', color='blue')
axs[1, 0].set_title('Sales USD - Yearly')
axs[1, 0].set_xlabel('Date')
axs[1, 0].set_ylabel('Sales_USD')
yearly_sales = df['Number_of_Sales'].resample('Y').sum()
axs[1, 1].plot(yearly_sales.index, yearly_sales, label='Number_of_Sales', color='green')
axs[1, 1].set_title('Number of Sales - Yearly')
axs[1, 1].set_xlabel('Date')
axs[1, 1].set_ylabel('Number_of_Sales')

plt.tight_layout()
plt.savefig('Yearly Data - Sales USD and Number of Sales.png')
plt.show()

# TREND ANALYSIS AND FORECASTING - Cumulative Metrics

# 1. ARIMA Model for Cumulative Sales USD
model1 = ARIMA(df1['Sales_USD_cumsum'], order=(5, 1, 0))
model_fit = model1.fit()
# Print model summary
print(model_fit.summary())
# Forecast the future values
forecast = model_fit.forecast(steps=10)
print("\nForecasted values:")
print(forecast)
# Plot the forecast
plt.plot(df1['Sales_USD_cumsum'], label='Original')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.savefig('Cumulative Sales USD - Original vs Forecast.png')
plt.show()

# 2. ARIMA Model for Cumulative Number of Sales
model2 = ARIMA(df1['Number_of_Sales_cumsum'], order=(5, 1, 0))
model_fit = model2.fit()
# Print model summary
print(model_fit.summary())
# Forecast the future values
forecast = model_fit.forecast(steps=10)
print("\nForecasted values:")
print(forecast)
# Plot the forecast
plt.plot(df1['Number_of_Sales_cumsum'], label='Original')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.savefig('Cumulative Number of Sales - Original vs Forecast.png')
plt.show()

# TREND ANALYSIS AND FORECASTING - Actual Metrics

# 1. ARIMA Model for Sales USD
model3 = ARIMA(df1['Sales_USD'], order=(5, 1, 0))
model_fit = model3.fit()
# Print model summary
print(model_fit.summary())
# Forecast the future values
forecast = model_fit.forecast(steps=10)
print("\nForecasted values:")
print(forecast)
# Plot the forecast
plt.plot(df1['Sales_USD'], label='Original')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.savefig('Actual Sales USD - Original vs Forecast.png')
plt.show()

# 2. ARIMA Model for Number of Sales
model4 = ARIMA(df1['Number_of_Sales'], order=(5, 1, 0))
model_fit = model4.fit()
# Print model summary
print(model_fit.summary())
# Forecast the future values
forecast = model_fit.forecast(steps=10)
print("\nForecasted values:")
print(forecast)
# Plot the forecast
plt.plot(df1['Number_of_Sales'], label='Original')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.savefig('Actual Number of Sales - Original vs Forecast.png')
plt.show()