
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from VanillaRNN import VanillaRNN
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler
from torch.nn.utils import clip_grad_norm_

def prepare_data(df_cols = ["TMAX"], seq_length = 10):
    df = pd.read_csv("weather.csv", parse_dates=["DATE"], index_col="DATE") 

    # data cleaning
    cols_to_fill = ['WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06','WT08', 'WT09'] 
    for col in cols_to_fill:
        if col in df.columns: 
            df[col] = df[col].fillna(0)

    # get only desired columns
    df = df[df_cols].dropna()

    # scale dataframe
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # get sequences and targets
    sequences, targets = [], []
    for i in range(len(df_scaled) - seq_length):
        sequences.append(df_scaled[i:i+seq_length])  
        targets.append(df_scaled[i+seq_length, 0]) 

    X, Y = np.array(sequences), np.array(targets)

    # split the data
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, shuffle=False)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, shuffle=False)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)  
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(-1) 
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(-1)  

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler, df

def train_and_test_model(config, prepare_data_fn = None):
    df_cols = config["df_cols"]
    seq_length = config["seq_length"]
    hidden_size = config["hidden_size"]
    epochs = config["num_epochs"]
    lr = config["learning_rate"]
    experiment_name = config["experiment_name"]

    input_size = len(df_cols)

    if prepare_data_fn is None:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler, df = prepare_data(df_cols, seq_length)
    else:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler, df = prepare_data_fn(df_cols, seq_length)
    
    model = VanillaRNN(hidden_size = hidden_size, input_size = input_size, output_size = 1)

    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    clip_value = 1.0

    # training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()

        # gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=clip_value)

        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, Y_val)
            val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'\tEpoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)

    # get predictions
    dummy_features = np.zeros((len(predictions), len(df_cols))) 
    dummy_features[:, 0] = predictions.numpy().flatten()  
    predictions_original_scale = scaler.inverse_transform(dummy_features)
    predicted_tmax = predictions_original_scale[:, 0]  # extract TMAX predictions

    # get actual tmax
    dummy_features_actual = np.zeros((len(Y_test), len(df_cols)))  
    dummy_features_actual[:, 0] = Y_test.numpy().flatten()  
    actual_original_scale = scaler.inverse_transform(dummy_features_actual)
    actual_tmax = actual_original_scale[:, 0]  # extract TMAX values
    
    print(f"Predicted: {predicted_tmax}")
    print(f"Actual: {actual_tmax}")

    test_dates = df.index[seq_length:][-len(Y_test):] 

    df_name = "alldata" if len(df_cols) > 1 else "TMAX"
    plot_actual_vs_pred_temps(test_dates, actual_tmax, predicted_tmax, f"images/{experiment_name}/df{df_name}_sl{seq_length}_hs{hidden_size}_lr{lr}_e{epochs}.png")

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



def plot_actual_vs_pred_temps(test_dates, actual_tmax, predicted_tmax, save_name):
    #test_dates = [datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d") for date in test_dates]

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actual_tmax, label="Actual TMAX", color="blue", linestyle="-")
    plt.plot(test_dates, predicted_tmax, label="Predicted TMAX", color="red", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°F)")
    plt.title("Boston High Temperature Prediction using Simple Predictor")
    plt.legend()
    plt.savefig(save_name)
    plt.close()

