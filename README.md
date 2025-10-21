# Fine_tuning_and_patching_raw_vibration
_Bearing Fault Diagnosis using Vibration Data & Transformer Fine-Tuning
_
**Overview**

This project implements bearing fault diagnosis using the Case Western Reserve University (CWRU) bearing dataset.
The notebook performs data preprocessing, feature extraction, and fine-tuning of a Transformer-based deep learning model to classify different bearing fault types based on vibration signals.

**Complete machine learning workflow: from raw signal handling to model evaluation:**

**Summary:**

Automated data extraction from .zip and .mat vibration files

Signal processing: normalization, segmentation, and noise filtering

Custom PyTorch Dataset & DataLoader for vibration sequences

Transformer-based model fine-tuning for fault classification

Performance evaluation using accuracy

**About the Dataset:**

Source: Case Western Reserve University Bearing Dataset (CWRU)

Format: .mat files containing time-domain vibration signals

Fault Types: Normal, Inner Race Fault, Outer Race Fault, Ball Fault, etc.

Sampling Frequency: 12 kHz (typical for CWRU)

The dataset is automatically extracted and processed inside the notebook.



**Full Technical Workflow:**

_1. Data Preparation and dataset creation_

Mounts Google Drive and unzips the dataset

Reads .mat vibration files using scipy.io.loadmat

Normalizes and segments signals into uniform-length samples

Creates labeled training and validation splits using train_test_split

Converts preprocessed vibration signals into a torch.utils.data.Dataset

Encodes class labels using ClassLabel from Hugging Face Datasets

Combines samples into a DataLoader

_2. Model Architecture and training_

Implements a Transformer encoder network using PyTorch & Hugging Face Transformers

Input embeddings capture vibration temporal patterns

Fine-tuning performed with Adam optimizer and cross-entropy loss

Configurable hyperparameters (epochs, learning rate, batch size)

_3. Evaluation & Results_

Computes classification metrics

Saves trained model checkpoints for later inference
