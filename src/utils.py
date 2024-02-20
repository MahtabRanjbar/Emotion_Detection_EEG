import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def save_confusion_matrix(y_true, y_pred, model_type, output_dir="."):
    conf_matrix = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d")
    plt.title(f"Confusion Matrix for {model_type}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{output_dir}/{model_type}_confusion_matrix.png")
    plt.close()


def save_classification_report(y_true, y_pred, model_type, output_dir="."):
    report = classification_report(y_true.cpu().numpy(), y_pred.cpu().numpy())
    with open(f"{output_dir}/{model_type}_classification_report.txt", "w") as f:
        f.write(report)


def plot_accuracy_loss(
    train_losses,
    train_accuracies,
    val_losses,
    val_accuracies,
    model_type,
    output_dir=".",
):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, len(train_accuracies) + 1), train_accuracies, label="Training Accuracy"
    )
    plt.plot(
        range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{output_dir}/{model_type}_train_val_hsitory.png")
    plt.tight_layout()
    plt.show()
