import torch
from torch import nn as nn
import torch.nn.functional as F
from .preprocessing import txt_to_tensor, tensor_to_txt, create_vocab


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, vocab, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab = vocab
        self.device = device

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embedding_dim)
        
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers)

        self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, prev_state):
        x = self.embedding(x)
        x, state = self.lstm(x, prev_state)
        x = self.linear(self.dropout(x))
        return x, state
    
    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size),
                torch.zeros(self.num_layers, sequence_length, self.hidden_size))
    
    def generate_by_names(self, input, max_new_names):
        names_generated = 0
        (hidden, cell) = self.init_state(len(input))
        hidden = hidden.to(self.device)
        cell = cell.to(self.device)
        x = txt_to_tensor(input, self.vocab, self.device).to(self.device)
        while names_generated < max_new_names:
            x = x.reshape(1, x.shape[0])
            output, (hidden, cell) = self.forward(x[:, -len(input):], (hidden, cell))
            output = output[:, -1, :].squeeze()
            softmax = F.softmax(output, dim=-1)
            x_next = torch.multinomial(softmax, num_samples=1).reshape(1)
            if x_next[0] == 0:
                names_generated += 1
            if x.size()[1] == 1:
                x = torch.cat((x.reshape(1), x_next))
            else: 
                x = torch.cat((x.squeeze(), x_next))

        return tensor_to_txt(x, self.vocab)
    
    def generate_with_letter(self, input, max_new_names):
        x = ""
        for _ in range(max_new_names):
            x += self.generate_by_names(input, 1)
            
        return x