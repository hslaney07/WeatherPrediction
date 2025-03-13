import os
import csv
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from train import train_and_test_model

'''
For this, we need to prepare data differently than the other configs.
We need to use the weather.csv with info from 1/1/2020 to 3/5/2025 and recent_weather.csv with information from 3/6/2025 - 3/10/2025
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
    X_test = X[-num_test_dates:]
    Y_test = Y[-num_test_dates:]

    X_train_val = X[:-num_test_dates]
    Y_train_val = Y[:-num_test_dates]

    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.3, shuffle=False)
    
    # convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1) 
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(-1)  
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(-1) 

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler, df_combined

# set up configuration - best performing configuration from the second experiment
config = {
        "df_cols": ["TMAX", "AWND", "TAVG", "TMIN", "WT01", 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08', 'WT09'],
        "seq_length": 40,
        "hidden_size": 150,
        "num_layers": 2,
        "num_epochs": 200,
        "learning_rate": 0.001, 
        "experiment_name": "recent_data"
    }

# before running training - make sure directories exist that are needed during training
os.makedirs("experiment_results", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("images/recent_data", exist_ok=True)

csv_file = "experiment_results/recent_weather.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Features", "Seq Length", "Hidden Size", "Num Layers", "Num Epochs", "Learning Rate",
        "Train Loss", "Val Loss", "Test Loss (MSE)", "Test Error (MAE)",
        "Worst Guess Actual", "Worst Guess Predicted", "Worst Guess Error", "Worst Guess Date"
    ])

   
    print(f"Running experiment with config: {config}")
    result = train_and_test_model(config, prepare_data)

    # write results to the CSV file
    writer.writerow([
        ", ".join(config["df_cols"]), config["seq_length"], config["hidden_size"], config["num_layers"], config["num_epochs"], config["learning_rate"],
        result["last_train_loss"], result["last_val_loss"], result["test_loss_mse"], result["test_error_mae"],
        result["worst_guess"]["actual"], result["worst_guess"]["predicted"], result["worst_guess"]["error"], result["worst_guess"]["date"]
    ])

print(f"Experiment completed. Results saved to {csv_file}.")


