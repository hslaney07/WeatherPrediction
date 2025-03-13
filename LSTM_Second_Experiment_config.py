import csv
import itertools
import os
from train import train_and_test_model

# setting up configurations for this experiment
df_cols = [
            ["TMAX", "AWND", "TAVG", "TMIN", "WT01", 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08', 'WT09'],
           ]

seq_lengths = [10, 30, 40]
hidden_sizes = [100, 150]
num_layers_list = [2, 6, 10]
epochs_list = [200, 250, 300]
lrs = [0.001]


all_configs = [
    {
        "df_cols": df_cols,
        "seq_length": seq_length,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "experiment_name": "second_experiment"
    }
    for df_cols, seq_length, hidden_size, num_layers, num_epochs, lr in itertools.product(
        df_cols, seq_lengths, hidden_sizes, num_layers_list, epochs_list, lrs
    )
]

# before running training - make sure directories exist that are needed during training
os.makedirs("experiment_results", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("images/second_experiment", exist_ok=True)

csv_file = "experiment_results/second_experiment_results.csv"

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Features", "Seq Length", "Hidden Size", "Num Layers", "Num Epochs", "Learning Rate",
        "Train Loss", "Val Loss", "Test Loss (MSE)", "Test Error (MAE)",
        "Worst Guess Actual", "Worst Guess Predicted", "Worst Guess Error", "Worst Guess Date"
    ])

    # run experiments
    for config in all_configs:
        print(f"Running experiment with config: {config}")
        result = train_and_test_model(config)

        # write results to the CSV file
        writer.writerow([
            ", ".join(config["df_cols"]), config["seq_length"], config["hidden_size"], config["num_layers"], config["num_epochs"], config["learning_rate"],
            result["last_train_loss"], result["last_val_loss"], result["test_loss_mse"], result["test_error_mae"],
            result["worst_guess"]["actual"], result["worst_guess"]["predicted"], result["worst_guess"]["error"], result["worst_guess"]["date"]
        ])

print(f"All experiments completed. Results saved to {csv_file}.")