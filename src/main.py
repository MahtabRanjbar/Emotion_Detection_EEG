import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src.data_loader import EEGDataset, create_data_loaders
from src.evaluate import evaluate, test_model
from src.model.gru import GRUModel
from src.model.lstm import LSTMModel
from src.train import train_model
from src.utils import (plot_accuracy_loss, save_classification_report,
                       save_confusion_matrix)

os.makedirs("../report", exist_ok=True)
# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {"GRU": GRUModel, "LSTM": LSTMModel}

input_size = 2548
hidden_size = 256
num_layers = 4
num_classes = 3
learning_rate = 0.001
batch_size = 64
num_epochs = 2
csv_file = "../data/emotions.csv"


def run_experiment(model_type):

    # Assuming create_data_loaders is a function you have defined elsewhere
    train_loader, val_loader, test_loader = create_data_loaders(csv_file, batch_size)

    model = models[model_type](input_size, hidden_size, num_layers, num_classes).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        criterion,
        optimizer,
        num_epochs,
        model_type,
        "../report",
    )

    _, y_pred, y_true = test_model(
        model, model_type, "../report", test_loader, device, criterion
    )
    save_confusion_matrix(y_true, y_pred, model_type, output_dir="../report")
    save_classification_report(y_true, y_pred, model_type, output_dir="../report")

    # savinf teaining history
    plot_accuracy_loss(
        history["train_loss"],
        history["train_accuracy"],
        history["val_loss"],
        history["val_accuracy"],
        model_type,
        output_dir="../report",
    )


def main():
    for model_type in models.keys():
        run_experiment(model_type)


if __name__ == "__main__":
    # This block of code will only run if the file is executed directly
    main()
