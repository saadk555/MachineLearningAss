import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the dataset 
df = pd.read_csv('shampoo_sales.csv', index_col='Month', parse_dates=True)

# Visualize the time series
plt.figure(figsize=(12, 6))
plt.plot(df['Sales'])
plt.title('Shampoo Sales Time Series')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

# Check stationarity (ADF test)
result = adfuller(df['Sales'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Build and fit the ARIMA model (you might need to experiment with the order)
model = ARIMA(df['Sales'], order=(1, 1, 1))  # First-order differencing (d=1)
model_fit = model.fit()
print(model_fit.summary())

# Evaluate the model (check residuals)
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(kind='kde')
plt.show()

# Make predictions
forecast = model_fit.forecast(steps=12)  # Predict the next 12 months
print(forecast)

mae = mean_absolute_error(df['Sales'][1:], model_fit.fittedvalues[1:])  # Skip the first value due to differencing
mse = mean_squared_error(df['Sales'][1:], model_fit.fittedvalues[1:])
rmse = np.sqrt(mse)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

