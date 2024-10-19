import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from datetime import timedelta

# Read the CSV file
data = pd.read_csv("c:\\Users\\DELL\\Downloads\\tatasteel.csv")
data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
TATASTEEL = data['Close Price'].values
DATE = data['Date']

# Calculate log returns
returns = np.diff(np.log(TATASTEEL))

# Define and fit EGARCH model
model = arch_model(returns, vol='EGARCH', p=1, q=1, mean='AR', lags=1, dist='t')
results = model.fit(disp='off')

# Print model summary
print(results.summary())

# Calculate fitted returns
fitted_returns = results.params['Const'] + results.params['y[1]'] * np.roll(returns, 1)
fitted_returns[0] = results.params['Const']  # First value can't use previous return

# Calculate RMSE
rmse = np.sqrt(np.mean((returns - fitted_returns)**2))
print(f"RMSE: {rmse}")

# Forecasting
forecast_horizon = 10
forecasts = results.forecast(horizon=forecast_horizon, method='simulation', simulations=10000)

forecasted_mean = forecasts.mean.iloc[-1]
forecasted_vol = np.sqrt(forecasts.variance.iloc[-1])

# Convert forecasts back to price levels
last_price = TATASTEEL[-1]
predicted_prices = last_price * np.exp(np.cumsum(forecasted_mean))

# Predict close price values for next 10 days
pred_dates = [DATE.iloc[-1] + timedelta(days=i+1) for i in range(forecast_horizon)]

print("\nPredicted Close Prices for next 10 days:")
for date, price in zip(pred_dates, predicted_prices):
    print(f"{date.date()}: {price:.2f}")

# Calculate confidence intervals
ci_lower = last_price * np.exp(np.cumsum(forecasted_mean - 1.96 * forecasted_vol))
ci_upper = last_price * np.exp(np.cumsum(forecasted_mean + 1.96 * forecasted_vol))

print("\n95% Confidence Intervals for Predicted Prices:")
for date, lower, upper in zip(pred_dates, ci_lower, ci_upper):
    print(f"{date.date()}: ({lower:.2f}, {upper:.2f})")

# Plotting the predictions
plt.figure(figsize=(12, 6))
plt.plot(DATE, TATASTEEL, label='Historical')
plt.plot(pred_dates, predicted_prices, label='Forecast')
plt.fill_between(pred_dates, ci_lower, ci_upper, color='red', alpha=0.2)
plt.title('Tata Steel Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()