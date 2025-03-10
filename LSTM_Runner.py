import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from WeatherLSTM import WeatherLSTM

### LOAD AND PREPARE THE DATA ###############################################
df = pd.read_csv("weather.csv", parse_dates=["DATE"], index_col="DATE") 
df = df[["TMAX"]].dropna()

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

def create_sequences(data, seq_length=30):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

SEQ_LENGTH = 10  # 10 days of past data to predict the next day's high temp

X, Y = create_sequences(df_scaled, SEQ_LENGTH)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float32).view(X_train.shape[0], SEQ_LENGTH, 1)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).view(X_test.shape[0], SEQ_LENGTH, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

######### Train the model #############################################################
model = WeatherLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

model.eval()
with torch.no_grad():
    predictions = model(X_test)

predictions = scaler.inverse_transform(predictions.numpy())
actual = scaler.inverse_transform(Y_test.numpy())

# plot results with dates
test_dates = df.index[SEQ_LENGTH:][-len(Y_test):] 

plt.figure(figsize=(12, 6))
plt.plot(test_dates, actual, label="Actual TMAX", color="blue")
plt.plot(test_dates, predictions, label="Predicted TMAX", color="red", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Boston High Temperature Prediction using LSTM")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()