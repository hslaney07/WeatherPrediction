import csv
import itertools
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from train import train_and_test_model

def prepare_data():
    # prepare the data including recent weather data
    df_train_val = pd.read_csv("weather.csv", parse_dates=["DATE"])
    df_test = pd.read_csv("recent_weather.csv", parse_dates=["DATE"])

    df = pd.concat([df_train_val, df_test], axis=0)

    # get desired columns
    df = df[["DATE", "TMAX"]].dropna()

    # extract the day of the year (0-364.2422)
    df['DayOfYear'] = df['DATE'].dt.dayofyear - 1  

    def is_leap_year(year):
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    df["Year"] = df["DATE"].dt.year  # Extract year
    df["TotalDays"] = df["Year"].apply(lambda y: 366 if is_leap_year(y) else 365)  # Get total days in year

    df["DayOfYear"] = (df["DayOfYear"] * 365) // df["TotalDays"] 

    # normalize TMAX
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df['TMAX'] = scaler.fit_transform(df[['TMAX']])

    num_test_dates = df_test.shape[0]
    train = df[:-num_test_dates]
    test = df[-num_test_dates:]

    # get train, val and test
    X_train_val = train['DayOfYear'].values
    Y_train_val = train['TMAX'].values

    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.3, shuffle=False)

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    Y_val = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(test["DayOfYear"].values, dtype=torch.float32).unsqueeze(1)
    Y_test = torch.tensor(test["TMAX"].values, dtype=torch.float32).unsqueeze(1)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, df, scaler

# setting up configurations for this experiment
epochs_list = [700]
lrs = [0.01]

all_configs = [
    {
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "experiment_name": "Predictor_On_Recent_Data"
    }
    for num_epochs, lr in itertools.product(
        epochs_list, lrs
    )
]

# before running training - make sure directories exist that are needed during training
os.makedirs("experiment_results", exist_ok=True)
os.makedirs("experiment_results/Simple", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("images/Predictor_On_Recent_Data", exist_ok=True)

csv_file = "experiment_results/Simple/Predictor_On_Recent_Data.csv"

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Num Epochs", "Learning Rate",
        "Train Loss", "Val Loss", "Test Loss (MSE)", "Test Error (MAE)",
    ])

    # run experiments
    for config in all_configs:
        print(f"Running experiment with config: {config}")
        result = train_and_test_model(config, prepare_data)

        # write results to the CSV file
        writer.writerow([
            config["num_epochs"], config["learning_rate"],
            result["last_train_loss"], result["last_val_loss"], result["test_loss_mse"], result["test_error_mae"],
        ])

print(f"All experiments completed. Results saved to {csv_file}.")