import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from SimplePredictor import SinePredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def prepare_data():
    # load the data
    df = pd.read_csv("weather.csv", parse_dates=["DATE"])

    # get desired columns
    df = df[["DATE", "TMAX"]].dropna()

    # extract the day of the year (0-364.2422)
    df['DayOfYear'] = df['DATE'].dt.dayofyear - 1

    # leap years 
    leap_years = [2020, 2024]
    df['DayOfYear'] = df.apply(
        lambda row: row['DayOfYear'] - 1 if row['DATE'].year in leap_years and row['DayOfYear'] >= 60 else row['DayOfYear'],
        axis=1
    )

    # normalize DayOfYear to 0-2π for the sine function
    df['DayOfYear'] = df['DayOfYear'] / 365.2422

    # normalize TMAX
    scaler = MinMaxScaler(feature_range=(-1, 1))
    #scaler = StandardScaler()
    df['TMAX'] = scaler.fit_transform(df[['TMAX']])

    # set up train, val, and test data
    train_data = df[df['DATE'] <= '2023-12-31']

    val_data = df[(df['DATE'] >= '2024-01-01') & (df['DATE'] <= '2024-12-31')]
    test_data = df[df['DATE'] >= '2025-01-01']

    X_train = train_data['DayOfYear'].values
    Y_train = train_data['TMAX'].values

    X_val = val_data['DayOfYear'].values
    Y_val = val_data['TMAX'].values

    X_test = test_data['DayOfYear'].values
    Y_test = test_data['TMAX'].values

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    Y_val = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, df, scaler


def train_and_test_model(config, prepare_data_fn=None):
    epochs = config["num_epochs"]
    lr = config["learning_rate"]
    experiment_name = config["experiment_name"]

    if prepare_data_fn is not None:
         X_train, Y_train, X_val, Y_val, X_test, Y_test, df, scaler = prepare_data_fn()
    else:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, df, scaler = prepare_data()
    
    model = SinePredictor()
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

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, Y_val)
            val_losses.append(val_loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

        # plot on the last epoch
        if epoch == epochs - 1:
            with torch.no_grad():
                train_predictions = model(X_train).numpy()

            train_predictions = scaler.inverse_transform(train_predictions)
            Y_train_original = scaler.inverse_transform(Y_train.numpy().reshape(-1, 1))

            train_dates = df["DATE"][:len(Y_train)].values

            save_name = f"images/{experiment_name}/training_lr{lr}_e{epochs}.png"
            plot_actual_vs_pred_temps(train_dates, Y_train_original, train_predictions, save_name=save_name)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)

    predictions = predictions.numpy()
    Y_test = Y_test.numpy()
    
    predictions = scaler.inverse_transform(predictions)
    Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

    print(f"Predicted: {predictions}")
    print(f"Actual: {Y_test}")

    mae = mean_absolute_error(Y_test, predictions)
    mse = mean_squared_error(Y_test, predictions)

    test_dates = df["DATE"][-len(Y_test):].values
    save_name = f"images/{experiment_name}/lr{lr}_e{epochs}.png"
    plot_actual_vs_pred_temps(test_dates, Y_test, predictions, save_name = save_name)

    return {
        "config": config,
        "last_train_loss": train_losses[-1],
        "last_val_loss": val_losses[-1],
        "test_loss_mse": mse,
        "test_error_mae": mae,
    }


def plot_actual_vs_pred_temps(test_dates, actual_tmax, predicted_tmax, save_name):
    #test_dates = [datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d") for date in test_dates]

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actual_tmax, label="Actual TMAX", color="blue", marker='o', linestyle="-")
    plt.plot(test_dates, predicted_tmax, label="Predicted TMAX", color="red", marker='s', linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°F)")
    plt.title("Boston High Temperature Prediction using Simple Predictor")
    plt.legend()
    plt.savefig(save_name)
    plt.close()

