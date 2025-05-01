# your code here
import torch
import torch.nn as nn
from bpe_tokenizer import BPETokenizer
# your code here

class LSTM(nn.Module):
    def __init__(self, vocab_size, gate_f=True, gate_i=True, gate_o=True,  embedding_dim=128, hidden_dim=256, num_classes=2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.gate_f = gate_f
        self.gate_i = gate_i
        self.gate_o = gate_o
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.W_f = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))

        self.W_i = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))

        self.W_o = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))

        self.W_c = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.b_c = nn.Parameter(torch.zeros(hidden_dim))

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hidden_state=None):
        batch_size = x.size(0)

        if hidden_state is None:
            h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        else:
            h_t, c_t = hidden_state

        embed = self.embedding(x)

        for t in range(x.size(1)):
            mask = (x[:, t] != 0).float().unsqueeze(1)
            x_t = embed[:, t, :]

            combined = torch.cat((h_t, x_t), dim=1)

            ones = 0

            if not self.gate_f or not self.gate_i or  not self.gate_o:
              ones = torch.zeros(self.hidden_dim) + 1
              ones = ones.to(x.device)
            if self.gate_f:
              f_t = torch.sigmoid(self.W_f(combined) + self.b_f)
            else:
              f_t = ones

            if self.gate_i:
              i_t = torch.sigmoid(self.W_i(combined) + self.b_i)
            else:
              i_t = ones

            if self.gate_o:
              o_t = torch.sigmoid(self.W_o(combined) + self.b_o)
            else:
              o_t = ones
            c_tilde = torch.tanh(self.W_c(combined) + self.b_c)

            c_new = f_t * c_t + i_t * c_tilde
            c_t = c_new * mask + c_t * (1 - mask)
            h_new = o_t * torch.tanh(c_t)
            h_t = h_new * mask + h_t * (1 - mask)

        output = self.fc(h_t)

        return output
    


class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, gate_f=True, gate_i=True, gate_o=True):
        super(LSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.gate_f = gate_f
        self.gate_i = gate_i
        self.gate_o = gate_o

        self.W_f = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_i = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.b_f = nn.Parameter(torch.zeros(hidden_dim))
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))
        self.b_c = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h_t, c_t, mask):
        combined = torch.cat((h_t, x), dim=1)
        ones = torch.ones_like(h_t)

        f_t = torch.sigmoid(self.W_f(combined) + self.b_f) if self.gate_f else ones
        i_t = torch.sigmoid(self.W_i(combined) + self.b_i) if self.gate_i else ones
        o_t = torch.sigmoid(self.W_o(combined) + self.b_o) if self.gate_o else ones

        c_tilde = torch.tanh(self.W_c(combined) + self.b_c)
        c_new = f_t * c_t + i_t * c_tilde
        c_t = c_new * mask + c_t * (1 - mask)

        h_new = o_t * torch.tanh(c_t)
        h_t = h_new * mask + h_t * (1 - mask)

        return h_t, c_t

class MultiLayerLSTM(nn.Module):
    def __init__(self, vocab_size, num_layers=1, gate_f=True, gate_i=True, gate_o=True,
                 embedding_dim=128, hidden_dim=256, num_classes=2):
        super(MultiLayerLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm_layers = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = embedding_dim if layer == 0 else hidden_dim
            self.lstm_layers.append(
                LSTMLayer(input_dim, hidden_dim, gate_f, gate_i, gate_o)
            )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hidden_states=None):
        batch_size = x.size(0)

        if hidden_states is None:
            hidden_states = []
            for _ in range(self.num_layers):
                h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
                c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
                hidden_states.append((h_t, c_t))

        embed = self.embedding(x)

        for t in range(x.size(1)):
            x_t = embed[:, t, :]
            mask = (x[:, t] != tokenizer.pad_token_id).float().unsqueeze(1) 

            new_hidden_states = []
            for layer in range(self.num_layers):
                h_t, c_t = hidden_states[layer]
                if layer > 0:
                    x_t = h_prev_layer

                h_t, c_t = self.lstm_layers[layer](x_t, h_t, c_t, mask)
                new_hidden_states.append((h_t, c_t))
                h_prev_layer = h_t

            hidden_states = new_hidden_states

        last_hidden = hidden_states[-1][0]
        output = self.fc(last_hidden)

        return output
    


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, gate_f=True, gate_i=True, gate_o=True, embedding_dim=128, hidden_dim=256, num_classes=2):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.gate_f = gate_f
        self.gate_i = gate_i
        self.gate_o = gate_o

        self.embedding = nn.Embedding(vocab_size, embedding_dim)


        self.W_f = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))

        self.W_i = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))

        self.W_o = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))

        self.W_c = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.b_c = nn.Parameter(torch.zeros(hidden_dim))

        self.W_f_bw = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.b_f_bw = nn.Parameter(torch.zeros(hidden_dim))

        self.W_i_bw = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.b_i_bw = nn.Parameter(torch.zeros(hidden_dim))

        self.W_o_bw = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.b_o_bw = nn.Parameter(torch.zeros(hidden_dim))

        self.W_c_bw = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.b_c_bw = nn.Parameter(torch.zeros(hidden_dim))

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, hidden_state=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        if hidden_state is None:
            h_t_fw = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c_t_fw = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            h_t_bw = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c_t_bw = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        else:
            h_t_fw, c_t_fw, h_t_bw, c_t_bw = hidden_state

        embed = self.embedding(x)

        forward_outputs = []
        for t in range(seq_len):
            x_t = embed[:, t, :]
            mask = (x[:, t] != 0).float().unsqueeze(1)

            combined_fw = torch.cat((h_t_fw, x_t), dim=1)

            ones = torch.zeros(self.hidden_dim).to(x.device) + 1

            f_t_fw = ones if not self.gate_f else torch.sigmoid(self.W_f(combined_fw) + self.b_f)
            i_t_fw = ones if not self.gate_i else torch.sigmoid(self.W_i(combined_fw) + self.b_i)
            o_t_fw = ones if not self.gate_o else torch.sigmoid(self.W_o(combined_fw) + self.b_o)

            c_tilde_fw = torch.tanh(self.W_c(combined_fw) + self.b_c)

            c_new_fw = f_t_fw * c_t_fw + i_t_fw * c_tilde_fw
            c_t_fw = c_new_fw * mask + c_t_fw * (1 - mask)

            h_new_fw = o_t_fw * torch.tanh(c_t_fw)
            h_t_fw = h_new_fw * mask + h_t_fw * (1 - mask)

            forward_outputs.append(h_t_fw)

        backward_outputs = []
        for t in reversed(range(seq_len)):
            x_t = embed[:, t, :]
            mask = (x[:, t] != 0).float().unsqueeze(1)

            combined_bw = torch.cat((h_t_bw, x_t), dim=1)

            ones = torch.zeros(self.hidden_dim).to(x.device) + 1

            f_t_bw = ones if not self.gate_f else torch.sigmoid(self.W_f_bw(combined_bw) + self.b_f_bw)
            i_t_bw = ones if not self.gate_i else torch.sigmoid(self.W_i_bw(combined_bw) + self.b_i_bw)
            o_t_bw = ones if not self.gate_o else torch.sigmoid(self.W_o_bw(combined_bw) + self.b_o_bw)

            c_tilde_bw = torch.tanh(self.W_c_bw(combined_bw) + self.b_c_bw)

            c_new_bw = f_t_bw * c_t_bw + i_t_bw * c_tilde_bw
            c_t_bw = c_new_bw * mask + c_t_bw * (1 - mask)


            h_new_bw = o_t_bw * torch.tanh(c_t_bw)
            h_t_bw = h_new_bw * mask + h_t_bw * (1 - mask)

            backward_outputs.insert(0, h_t_bw) 

        final_forward = forward_outputs[-1]
        final_backward = backward_outputs[0]

        combined_output = torch.cat((final_forward, final_backward), dim=1)

        output = self.fc(combined_output)

        return output