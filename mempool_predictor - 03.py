import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from math import sqrt
import uuid

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
                return {"transaction_count": data['n_tx'], "source": url}
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
        return response.json(), url
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None, None

# Function to preprocess and normalize data
def preprocess_data(data, is_fees_data=False):
    current_bitcoin_price = 58643  # Updated: Use current Bitcoin price based on analysis
    if is_fees_data:
        transaction_count = None  # No transaction count available
        fee_rate = data['fastestFee']  # Using fastest fee as the fee rate
        block_size = 1  # Assuming block size is 1 MB for simplification
        total_fees = 0  # Not available in fee data, so using 0
        external_factors = 1  # Placeholder for external factors (e.g., sentiment)
    else:
        transaction_count = data['transaction_count']  # Number of transactions in the mempool
        fee_rate = np.mean(average_fees)  # Using the pre-made list of average fees as a fallback
        block_size = 1  # Assuming block size is 1 MB for simplification
        total_fees = data.get('total_fee', 0)
        external_factors = 1  # Placeholder for external factors (e.g., sentiment)
    
    return np.array([transaction_count, fee_rate, block_size, current_bitcoin_price, total_fees, external_factors])

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

# Function to save the output to a file
def save_to_file(content):
    unique_id = uuid.uuid4().hex  # Generate a unique ID for the filename
    filename = f"projections_{unique_id}.md"  # Save as Markdown for formatted text
    with open(filename, 'w') as file:
        file.write(content)
    print(f"Results saved to {filename}")

# Main function to execute predictions and error calculations
def main():
    global transaction_volumes  # Use the global variable to store volumes over time
    mempool_data, mempool_source = fetch_mempool_data(), None
    is_fees_data = False
    
    if not mempool_data:
        print("Fetching fee data as fallback...")
        mempool_data, mempool_source = fetch_recommended_fees()
        is_fees_data = True
    
    if mempool_data:
        processed_data = preprocess_data(mempool_data, is_fees_data)
        transaction_volume = processed_data[0]  # This is the latest transaction volume
        
        if transaction_volume is not None:
            # Append the new transaction volume to the time series
            transaction_volumes.append(transaction_volume)
        
            # Only train ARIMA if we have a valid sequence of transaction volumes
            if len(transaction_volumes) > 1:
                arima_model = train_arima(transaction_volumes)
                if arima_model:
                    next_transaction_volume_prediction = arima_model.forecast(steps=1)[0]
                    plot_predictions(transaction_volumes, [next_transaction_volume_prediction], "Transaction Volume Prediction")
                else:
                    next_transaction_volume_prediction = None
            else:
                next_transaction_volume_prediction = None
        else:
            next_transaction_volume_prediction = None
        
        # Ensure next_transaction_volume_prediction is initialized before proceeding
        if next_transaction_volume_prediction is not None:
            print(f"Predicted Transaction Volume: {next_transaction_volume_prediction}")
        else:
            print("Transaction Volume Prediction: N/A (Insufficient data)")
        
        # Train Random Forest model for fee rate prediction
        feature_vector = np.array([processed_data[:-1]])  # Excluding the fee rate from the features
        target_fee_rate = np.array([processed_data[1]])  # Fee rate as the target
        rf_model = train_random_forest(feature_vector, target_fee_rate)
        
        # Predict fee rate for the next time step
        next_fee_rate_prediction = rf_model.predict(feature_vector)[0]
        
        # Example for historical predictions
        current_transaction_volume = processed_data[0]
        current_fee_rate = processed_data[1]
        
        # Calculate errors if ARIMA prediction is available
        if next_transaction_volume_prediction is not None:
            mae, rmse, mape = calculate_errors([current_transaction_volume], [next_transaction_volume_prediction])
            print(f"Transaction Volume Prediction: {next_transaction_volume_prediction} (MAE: {mae}, RMSE: {rmse}, MAPE: {mape})")
        else:
            print("Transaction Volume Prediction: N/A (Insufficient data)")
        
        mae, rmse, mape = calculate_errors([current_fee_rate], [next_fee_rate_prediction])
        print(f"Fee Rate Prediction: {next_fee_rate_prediction} sat/byte (MAE: {mae}, RMSE: {rmse}, MAPE: {mape})")
        
        # Display projections with explanations
        output = "# Projections with Explanations\n\n"
        output += f"**Source of Today's Data:** {mempool_source}\n\n" if mempool_source else "**Source of Today's Data:** Data from fallback API\n\n"
        output += f"**Source of Previous Data (Last 10,000 Transactions):**\n- Example Fees: {average_fees}\n\n"
        
        time_periods = ["End of Day", "End of Month", "End of Year", "2 Years", "5 Years", "10 Years", "50 Years", "100 Years"]
        growth_rate_per_day = 0.005  # Example: Daily growth rate of 0.5%
        days = [1, 30, 365, 365*2, 365*5, 365*10, 365*50, 365*100]
        
        for period, day_count in zip(time_periods, days):
            if current_transaction_volume is not None:
                projected_transaction_count = current_transaction_volume * (1 + growth_rate_per_day) ** day_count
                output += f"## {period}:\n"
                output += f"  **Current Transaction Volume:** {current_transaction_volume}\n"
                output += f"  **Growth Rate:** {growth_rate_per_day} per day\n"
                output += f"  **Projected Transaction Volume:** {projected_transaction_count}\n\n"
            else:
                output += f"## {period}:\n  **Projection:** N/A (Insufficient data)\n\n"

        save_to_file(output)

if __name__ == "__main__":
    main()
