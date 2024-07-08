import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset_path = tf.keras.utils.get_file("aclImdb_v1.tar.gz", dataset_url, untar=True, cache_dir='.', cache_subdir='')

# Extracted dataset directory
dataset_dir = os.path.join(os.path.dirname(dataset_path), 'aclImdb')

# Load the data
def load_data(dataset_path, split='train'):
    data = []
    labels = []
    for label in ['pos', 'neg']:
        path = os.path.join(dataset_path, split, label)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                data.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    return data, labels

# Load data using the function
train_data, train_labels = load_data(dataset_dir, split='train')
test_data, test_labels = load_data(dataset_dir, split='test')

# Example of how to use the loaded data
print(f"Number of training samples: {len(train_data)}")
print(f"Number of test samples: {len(test_data)}")

# Rest of your code...
