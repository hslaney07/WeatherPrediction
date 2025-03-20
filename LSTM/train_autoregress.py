from datetime import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from WeatherLSTM import WeatherLSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

'''
Uses the weather.csv with info from 1/1/2020 to 3/5/2025 and recent_weather.csv with information from 3/6/2025 - 3/15/2025
However, for X_test - this is only a tensor of shape (1, seq_length, num_input).
   This tensor represents the historical data (past 40 days) leading up to the day we are attempting to forecast
'''
def prepare_data(df_cols=["TMAX"], seq_length=30):
    # load data
    df_train_val = pd.read_csv("weather.csv", parse_dates=["DATE"], index_col="DATE")
    df_test = pd.read_csv("recent_weather.csv", parse_dates=["DATE"], index_col="DATE")

    df_combined = pd.concat([df_train_val, df_test], axis=0) 

    # data cleaning 
    cols_to_fill = ['WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08', 'WT09', "AWND"] # For some reason AWND had some values as NaN which the training set doesnt
    for col in cols_to_fill:
        if col in df_combined.columns: 
            df_combined[col] = df_combined[col].fillna(0)

    df_combined = df_combined[df_cols].dropna()

    # scale data 
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_combined_scaled = scaler.fit_transform(df_combined)
   
    sequences, targets = [], []
    for i in range(len(df_combined_scaled) - seq_length):
        sequences.append(df_combined_scaled[i:i+seq_length])  
        targets.append(df_combined_scaled[i+seq_length, 0]) 

    X, Y = np.array(sequences), np.array(targets)

    # split the data
    num_test_dates = df_test.shape[0]
    X_test = X[-num_test_dates:][0]
    Y_test = Y[-num_test_dates:]

    X_train_val = X[:-num_test_dates]
    Y_train_val = Y[:-num_test_dates]

    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.3, shuffle=False)
    
    # convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1) 
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(-1)  
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(0)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(-1) 

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler, df_combined


# Autoregressive prediction function for testing data. We are predicting the weather for num_steps days.
def autoregressive_predict_testing(model, test_data, num_steps):
    model.eval()
    predictions = []
 
    with torch.no_grad():
        for _ in range(num_steps):
            # Predict the next temperature
            prediction = model(test_data)
            if prediction.shape == (1, 1):
                tmax_prediction = prediction.item()
                predictions.append(tmax_prediction)
                input = torch.zeros(12) 
                input[0] = tmax_prediction
                predicted_input = input.unsqueeze(0).unsqueeze(0)
            else:
                tmax_prediction = prediction[0][0].item()
                predictions.append(tmax_prediction)
                predicted_input = prediction.unsqueeze(1)
            
            # We will update the test day - remove the day 1 of the old input and add our predicted information for the previous day at the end.
            test_data = torch.cat(
                [test_data[:, 1:, :], predicted_input], dim=1
            )

    return np.array(predictions)

def train_and_test_model(config, prepare_data_fn = None):
    df_cols = config["df_cols"]
    seq_length = config["seq_length"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    epochs = config["num_epochs"]
    lr = config["learning_rate"]
    experiment_name = config["experiment_name"]
    num_output = config["num_output"]

    input_size = len(df_cols)

    if prepare_data_fn is None:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler, df = prepare_data(df_cols, seq_length)
    else:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler, df = prepare_data_fn(df_cols, seq_length)
    

    model = WeatherLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_output=num_output)

    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    # training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train) 
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # validation step
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, Y_val) 
            val_losses.append(val_loss.item())

        if (epoch+1) % 10 == 0:
            print(f'\tEpoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

    model.eval()
    with torch.no_grad():
        predictions = autoregressive_predict_testing(model, X_test, num_steps=len(Y_test))

    # get predictions
    dummy_features = np.zeros((len(predictions), len(df_cols))) 
    dummy_features[:, 0] = predictions
    predictions_original_scale = scaler.inverse_transform(dummy_features)
    predicted_tmax = predictions_original_scale[:, 0]  # extract TMAX predictions

    # get actual tmax
    if Y_test.shape[1] != 1:
        Y_test = Y_test[:, 0]
    dummy_features_actual = np.zeros((len(Y_test), len(df_cols)))  
    dummy_features_actual[:, 0] = Y_test.numpy().flatten()  
    actual_original_scale = scaler.inverse_transform(dummy_features_actual)
    actual_tmax = actual_original_scale[:, 0]  # extract TMAX values

    print(f"Predicted: {predicted_tmax}")
    print(f"Actual: {actual_tmax}")

    test_dates = df.index[seq_length:][-len(Y_test):] 

    df_name = "alldata" if len(df_cols) > 1 else "TMAX"
    plot_actual_vs_pred_temps(test_dates, actual_tmax, predicted_tmax, f"images/{experiment_name}/df{df_name}_sl{seq_length}_hs{hidden_size}_nl{num_layers}_lr{lr}_e{epochs}.png")

    # absolute errors
    absolute_errors = np.abs(actual_tmax - predicted_tmax)

    # get the worst actual, predicted, and error values
    worst_guess_index = np.argmax(absolute_errors)
    worst_actual = actual_tmax[worst_guess_index]  
    worst_predicted = predicted_tmax[worst_guess_index]  
    worst_error = absolute_errors[worst_guess_index]
    worst_date = test_dates[worst_guess_index]

    # test mse and mae
    test_loss_mse = np.mean((actual_original_scale - predictions_original_scale) ** 2)
    test_error_mae = np.mean(np.abs(actual_original_scale - predictions_original_scale))

    return {
        "config": config,
        "last_train_loss": train_losses[-1],
        "last_val_loss": val_losses[-1],
        "test_loss_mse": test_loss_mse,
        "test_error_mae": test_error_mae,
        "worst_guess": {
            "actual": worst_actual,
            "predicted": worst_predicted,
            "error": worst_error,
            "date": worst_date.strftime('%Y-%m-%d'),
        }
    }


def plot_actual_vs_pred_temps(test_dates, actual_tmax, predicted_tmax, name):
    test_dates = [datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d") for date in test_dates]

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actual_tmax, label="Actual TMAX", color="blue", marker='o', linestyle="-")
    plt.plot(test_dates, predicted_tmax, label="Predicted TMAX", color="red", marker='s', linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°F)")
    plt.title("Boston High Temperature Prediction using LSTM")
    plt.legend()
    plt.savefig(name)
    plt.close()
