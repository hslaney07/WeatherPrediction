import os
import csv
from train_autoregress import train_and_test_model

# set up configuration - best performing configuration from the second experiment
config = {
        "df_cols": ["TMAX", "AWND", "TAVG", "TMIN", "WT01", 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08', 'WT09'],
        "seq_length": 40,
        "hidden_size": 150,
        "num_layers": 2,
        "num_epochs": 200,
        "learning_rate": 0.001, 
        "experiment_name": "LSTM_Autogress_Model_Unchanged",
        "num_output": 1
    }

# before running training - make sure directories exist that are needed
os.makedirs("experiment_results", exist_ok=True)
os.makedirs("experiment_results/LSTM", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("images/LSTM_Autogress_Model_Unchanged", exist_ok=True)

# write results to csv
csv_file = "experiment_results/LSTM/LSTM_Autogress_Model_Unchanged.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Features", "Seq Length", "Hidden Size", "Num Layers", "Num Epochs", "Learning Rate",
        "Train Loss", "Val Loss", "Test Loss (MSE)", "Test Error (MAE)",
        "Worst Guess Actual", "Worst Guess Predicted", "Worst Guess Error", "Worst Guess Date"
    ])

    print(f"Running experiment with config: {config}")
    result = train_and_test_model(config)

    # write results to the CSV file
    writer.writerow([
        ", ".join(config["df_cols"]), config["seq_length"], config["hidden_size"], config["num_layers"], config["num_epochs"], config["learning_rate"],
        result["last_train_loss"], result["last_val_loss"], result["test_loss_mse"], result["test_error_mae"],
        result["worst_guess"]["actual"], result["worst_guess"]["predicted"], result["worst_guess"]["error"], result["worst_guess"]["date"]
    ])

print(f"Experiment completed. Results saved to {csv_file}.")


