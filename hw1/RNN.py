# your code here
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from bpe_tokenizer import BPETokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=2):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.b_x = nn.Parameter(torch.zeros(hidden_dim))
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))
        self.b_f = nn.Parameter(torch.zeros(num_classes))
        self.hidden_dim = hidden_dim
        self.W_xh = nn.Linear(embedding_dim, hidden_dim)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        embedded = self.embedding(x) 
        mask = (x != 0).float().unsqueeze(-1)

        for t in range(embedded.size(1)):
            x_t = embedded[:, t, :]
            step_mask = mask[:, t]
            h_new = torch.tanh(self.W_xh(x_t) + self.W_hh(hidden) + self.b_x + self.b_h)
            hidden = h_new * step_mask + hidden * (1 - step_mask)

        output = self.dropout(hidden)
        output = self.fc(hidden) + self.b_f
        return output