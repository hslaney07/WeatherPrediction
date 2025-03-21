import csv
import itertools
import os
from train import train_and_test_model

# setting up configurations for this experiment
epochs_list = [100, 300, 500, 700, 1000, 2000, 5000]
lrs = [0.001, 0.01, 0.0001]

all_configs = [
    {
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "experiment_name": "Predictor_Training"
    }
    for num_epochs, lr in itertools.product(
        epochs_list, lrs
    )
]

# before running training - make sure directories exist that are needed during training
os.makedirs("experiment_results", exist_ok=True)
os.makedirs("experiment_results/Simple", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("images/Predictor_Training", exist_ok=True)

csv_file = "experiment_results/Simple/Predictor_Training.csv"

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Num Epochs", "Learning Rate",
        "Train Loss", "Val Loss", "Test Loss (MSE)", "Test Error (MAE)",
        "Worst Guess Actual", "Worst Guess Predicted", "Worst Guess Error", "Worst Guess Date"
    ])

    # run experiments
    for config in all_configs:
        print(f"Running experiment with config: {config}")
        result = train_and_test_model(config)

        # write results to the CSV file
        writer.writerow([
            config["num_epochs"], config["learning_rate"],
            result["last_train_loss"], result["last_val_loss"], result["test_loss_mse"], result["test_error_mae"]
        ])

print(f"All experiments completed. Results saved to {csv_file}.")