import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.cuda import empty_cache

from src.data_loader import EEGDataset, create_data_loaders
from src.evaluate import evaluate, test_model
from utils import classification_report, confusion_matrix, plot_accuracy_loss


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    optimizer,
    epochs,
    model_type,
    output_dir="../report",
):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    # Assuming an evaluate function exists
    log_path = f"{output_dir}/{model_type}_training_log.txt"
    with open(log_path, "w") as log_file:
        for epoch in range(epochs):
            # Training and validation code here

            model.train()
            train_loss = 0
            train_correct = 0
            train_samples = 0
            for data, targets in train_loader:
                data = data.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                train_correct += (predictions == targets).sum().item()
                train_samples += targets.size(0)

            # Calculate training loss and accuracy
            train_loss /= len(train_loader)
            train_accuracy = (train_correct / train_samples) * 100

            # Validation
            val_loss, val_accuracy, _, _ = evaluate(
                model, val_loader, device, criterion
            )

            print(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
            )

            # Write epoch results to log_file
            log_file.write(
                f"Epoch {epoch+1}/{epochs}: Training Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}% ...\n"
            )

            # Store in history
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
    return history
