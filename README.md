
# Emotion Classification Using EEG Data

This repository contains the implementation of emotion classification models using Electroencephalogram (EEG) data. The project utilizes Gated Recurrent Units (GRU) and Long Short-Term Memory (LSTM) models implemented in PyTorch to analyze and classify emotional states based on EEG signal patterns. Our goal is to demonstrate the effectiveness of recurrent neural networks (RNNs) in interpreting time-series data for emotion recognition.


## Installation

1. Clone the repository:

   ```
   git clone https://github.com/MahtabRanjbar/Emotion_Detection_EEG.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Training the Model

    ```bash
    python src/main.py
    ```

## Dataset
This is a dataset of EEG brainwave data that. The data was collected from two people (1 male, 1 female) for 3 minutes per state - positive, neutral, negative. We used a Muse EEG headband which recorded the TP9, AF7, AF8 and TP10 EEG placements via dry electrodes. Six minutes of resting neutral data is also recorded. you can find this dataset in [kaggle](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions/data)

## Data Preparation

Place your EEG dataset in CSV format inside the `data/` directory. The dataset should have features as columns, with the last column representing the emotion labels.



## Results


The results of our emotion classification models are documented comprehensively to facilitate analysis and comparison between the GRU and LSTM models. For each model, we have generated and stored the following artifacts:

### Classification Report

- A detailed classification report for each model, including precision, recall, F1-score, and support for each class, can be found in the `report/` directory. Files are named as follows:
  - `GRU_classification_report.txt`
  - `LSTM_classification_report.txt`

### Confusion Matrix

- Confusion matrices are saved as images to visually represent the performance of each model across different classes. These can be found in the `report/` directory with the filenames:
  - `GRU_confusion_matrix.png`
  - `LSTM_confusion_matrix.png`

### Training Logs

- We have logged the training process, including loss and accuracy at each epoch for both training and validation phases. These logs provide insights into the learning process and can help identify overfitting or underfitting. Log files are available in the `report/` directory:
  - `GRU_training_log.txt`
  - `LSTM_training_log.txt`

### Training and Validation Accuracy per Epoch

- To visualize the training progress and model convergence, we have plotted the training and validation accuracy/loss for each epoch. These plots are crucial for understanding the model's behavior over time and are saved in the `report/` directory as:
  - `GRU_train_val_hsitory.png`
  - `LSTM_train_val_hsitory.png`



---


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



---
