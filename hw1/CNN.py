# your code here
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, num_classes=2):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.conv1 = nn.Conv1d(embedding_dim, 100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, 100, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(embedding_dim, 100, kernel_size=5, padding=1)

        self.fc = nn.Linear(300, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = self.embedding(x) 
        x = x.permute(0, 2, 1) 

        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)

        x2 = F.relu(self.conv2(x))
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)

        x3 = F.relu(self.conv3(x))
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)

        x = torch.cat((x1, x2, x3), dim=1) 
        x = self.dropout(x)
        x = self.fc(x)

        return x
