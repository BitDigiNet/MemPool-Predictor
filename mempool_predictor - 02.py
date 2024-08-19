import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from math import sqrt

# Initialize an empty list to store transaction volumes over time
transaction_volumes = []

# Pre-made list of average fees from the last 10,000 transactions (replace with actual data)
average_fees = [10, 12, 15, 8, 9, 11, 14, 10, 13, 7]  # Example fee rates in satoshis/byte

# Function to fetch data from multiple sources
def fetch_mempool_data():
    api_endpoints = [
        "https://api.blockchain.info/stats",  # Blockchain.com endpoint
        "https://api.blockcypher.com/v1/btc/main",  # BlockCypher endpoint
        # Add other endpoints as needed
    ]

    for url in api_endpoints:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx, 5xx)
            try:
                data = response.json()
            except ValueError:
                print(f"Failed to decode JSON from {url}. Response content: {response.text}")
                continue

            # Parse data according to the specific API structure
            if 'n_tx' in data:  # Example check for Blockchain.com and BlockCypher data
                return {"count": data['n_tx']}
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {url}: {e}")
            continue

    print("Failed to fetch transaction volume data from all sources.")
    return None

# Function to fetch recommended fees (as a fallback or additional data source)
def fetch_recommended_fees():
    url = "https://mempool.space/api/v1/fees/recommended"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

# Function to preprocess and normalize data
def preprocess_data(data, is_fees_data=False):
    P_t = 58643  # Example Bitcoin price (in USD)
    if is_fees_data:
        V_t = None  # No transaction count available
        F_t = data['fastestFee']  # Using fastest fee as the fee rate
        B_t = 1  # Assuming block size is 1 MB for simplification
        M_t = 0  # Not available in fee data, so using 0
        X_t = 1  # Placeholder for external factors (e.g., sentiment)
    else:
        V_t = data['count']  # Number of transactions in the mempool
        F_t = np.mean(average_fees)  # Using the pre-made list of average fees as a fallback
        B_t = 1  # Assuming block size is 1 MB for simplification
        M_t = data.get('total_fee', 0)
        X_t = 1  # Placeholder for external factors (e.g., sentiment)
    
    return np.array([V_t, F_t, B_t, P_t, M_t, X_t])

# Function to train ARIMA model for transaction volume prediction
def train_arima(data_series):
    if data_series is not None and len(data_series) > 1:  # Ensure there is enough data
        model = ARIMA(data_series, order=(5, 1, 0))
        model_fit = model.fit()
        return model_fit
    else:
        print("Insufficient data for ARIMA model.")
        return None

# Function to train Random Forest model for fee rate prediction
def train_random_forest(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train, y_train)
    return rf_model

# Function to calculate prediction errors
def calculate_errors(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, rmse, mape

# Function to plot the predictions
def plot_predictions(actual, predicted, title):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
# Main function to execute predictions and error calculations
def main():
    global transaction_volumes  # Use the global variable to store volumes over time
    mempool_data = fetch_mempool_data()
    is_fees_data = False
    
    if not mempool_data:
        print("Fetching fee data as fallback...")
        mempool_data = fetch_recommended_fees()
        is_fees_data = True
    
    if mempool_data:
        processed_data = preprocess_data(mempool_data, is_fees_data)
        V_t = processed_data[0]  # This is the latest transaction volume
        
        if V_t is not None:
            # Append the new transaction volume to the time series
            transaction_volumes.append(V_t)
        
            # Only train ARIMA if we have a valid sequence of transaction volumes
            if len(transaction_volumes) > 1:
                arima_model = train_arima(transaction_volumes)
                if arima_model:
                    V_t1_pred = arima_model.forecast(steps=1)[0]
                    plot_predictions(transaction_volumes, [V_t1_pred], "Transaction Volume Prediction")
                else:
                    V_t1_pred = None
            else:
                V_t1_pred = None
        else:
            V_t1_pred = None
        
        # Ensure V_t1_pred is initialized before proceeding
        if V_t1_pred is not None:
            print(f"Predicted Transaction Volume: {V_t1_pred}")
        else:
            print("Transaction Volume Prediction: N/A (Insufficient data)")
        
        # Train Random Forest model for fee rate prediction
        X_train = np.array([processed_data[:-1]])  # Excluding the fee rate from the features
        y_train = np.array([processed_data[1]])  # Fee rate as the target
        rf_model = train_random_forest(X_train, y_train)
        
        # Predict fee rate for the next time step
        F_t1_pred = rf_model.predict(X_train)[0]
        
        # Example for historical predictions
        V_t_actual = processed_data[0]
        F_t_actual = processed_data[1]
        
        # Calculate errors if ARIMA prediction is available
        if V_t1_pred is not None:
            mae, rmse, mape = calculate_errors([V_t_actual], [V_t1_pred])
            print(f"Transaction Volume Prediction: {V_t1_pred} (MAE: {mae}, RMSE: {rmse}, MAPE: {mape})")
        else:
            print("Transaction Volume Prediction: N/A (Insufficient data)")
        
        mae, rmse, mape = calculate_errors([F_t_actual], [F_t1_pred])
        print(f"Fee Rate Prediction: {F_t1_pred} sat/byte (MAE: {mae}, RMSE: {rmse}, MAPE: {mape})")
        
        # Display projections with explanations
        print("\nProjections with Explanations:")
        time_periods = ["End of Day", "End of Month", "End of Year", "2 Years", "5 Years", "10 Years", "50 Years", "100 Years"]
        growth_rate = 0.005  # Example: Daily growth rate of 0.5%
        days = [1, 30, 365, 365*2, 365*5, 365*10, 365*50, 365*100]
        
        for period, day_count in zip(time_periods, days):
            if V_t_actual is not None:
                projected_tx_count = V_t_actual * (1 + growth_rate) ** day_count
                print(f"Transaction count for {period}:")
                print(f"Current Transaction Volume (V_t): {V_t_actual}")
                print(f"Growth Rate: {growth_rate} per day")
                print(f"Projected Transaction Count: {projected_tx_count:.2f} after {day_count} days")
            else:
                projected_tx_count = "N/A"
            
            projected_total_fees = F_t_actual * (V_t_actual if V_t_actual is not None else 1) * (1 + growth_rate) ** day_count
            print(f"Fee Rate (F_t): {F_t_actual} satoshis/byte")
            print(f"Projected Total Fees: {int(projected_total_fees)} satoshis after {day_count} days")
    else:
        print("Failed to retrieve any data.")

if __name__ == "__main__":
    main()
