import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class EEGDataset(Dataset):
    """Dataset class for EEG data where inputs and labels are already separated."""

    def __init__(self, inputs, labels):
        """
        Args:
            inputs (numpy.ndarray): Input features.
            labels (numpy.ndarray): Targets/labels.
        """
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            sample (torch.Tensor): The input features as a tensor.
            label (torch.Tensor): The label as a tensor.
        """
        # Convert numpy arrays to torch tensors
        sample = torch.tensor(self.inputs[idx], dtype=torch.float)
        label = torch.tensor(
            self.labels[idx], dtype=torch.long
        )  # Assuming classification task

        return sample, label


def create_data_loaders(csv_file_path, batch_size=64, val_size=0.2, test_size=0.2):
    # Read the dataset
    data = pd.read_csv(csv_file_path)
    label_mapping = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    data["label"] = data["label"].replace(label_mapping)

    # Convert to numpy
    data_np = data.to_numpy()

    # Split the dataset into inputs and labels
    inputs = data_np[:, :-1]
    labels = data_np[:, -1].astype(int)

    # Splitting the dataset into training+validation and test sets
    inputs_train_val, inputs_test, labels_train_val, labels_test = train_test_split(
        inputs, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Splitting the training+validation set into individual training and validation sets
    inputs_train, inputs_val, labels_train, labels_val = train_test_split(
        inputs_train_val,
        labels_train_val,
        test_size=val_size / (1.0 - test_size),
        random_state=42,
        stratify=labels_train_val,
    )

    # Creating dataset instances for each subset
    train_dataset = EEGDataset(inputs=inputs_train, labels=labels_train)
    val_dataset = EEGDataset(inputs=inputs_val, labels=labels_val)
    test_dataset = EEGDataset(inputs=inputs_test, labels=labels_test)

    # Creating DataLoader instances for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Assuming you have defined the EEGDataset class as before
