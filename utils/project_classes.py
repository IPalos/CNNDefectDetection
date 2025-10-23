import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd


# Classes
class Sample:
    def __init__(self, filename, position, tensor, label, pseudo_label, pseudo_label_confidence):
        self.filename = filename
        self.position = position
        self.tensor = tensor
        self.label = label
        self.pseudo_label = pseudo_label
        self.pseudo_label_confidence = pseudo_label_confidence

    def __str__(self):
        return f"Sample(filename={self.filename}, position={self.position}, tensor={self.tensor}, label={self.label}, pseudo_label={self.pseudo_label}, pseudo_label_confidence={self.pseudo_label_confidence})"

    def to_dataframe(self):
        return pd.DataFrame({
            "filename": [self.filename],
            "position": [self.position],
            "label": [self.label],
            "pseudo_label": [self.pseudo_label],
            "pseudo_label_confidence": [self.pseudo_label_confidence]
        })
    
    def is_high_confidence(self, threshold=0.9):
        """Check if pseudo-label confidence exceeds the threshold."""
        return self.pseudo_label_confidence >= threshold

class SampleDataset(Dataset):
    def __init__(self, samples, device):
        """Initialize with a list of Sample objects."""
        self.samples = samples
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'tensor': torch.tensor(sample.tensor, dtype=torch.float32).to(self.device),
            'label': torch.tensor(sample.label if sample.label is not None else -1, dtype=torch.long).to(self.device),
            'pseudo_label': torch.tensor(sample.pseudo_label if sample.pseudo_label is not None else -1, dtype=torch.long).to(self.device),
            'pseudo_label_confidence': torch.tensor(sample.pseudo_label_confidence if sample.pseudo_label_confidence is not None else 0.0, dtype=torch.float32).to(self.device)
        }

    
    


# Define a custom dataset
class TensorDataset(Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]


# Define the CNN model
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)  # Reduced size
        self.fc2 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.reshape(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
