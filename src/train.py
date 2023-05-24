import torch
from torch import nn as nn
import torch.optim as optim
from tqdm import tqdm
from .model import Model
from .preprocessing import create_vocab


def train(loader, **kwargs):
    n_epochs = kwargs['n_epochs']
    min_loss = kwargs['min_loss']
    sequence_length = kwargs['seq_length']
    device = kwargs['device']

    model = Model(vocab_size=kwargs['vocab_size'],
              embedding_dim=kwargs['embedding_dim'],
              hidden_size=kwargs['hidden_size'],
              dropout=kwargs['dropout'],
              num_layers=kwargs['num_layers'],
              vocab=create_vocab(kwargs['alphabet']),
              device=kwargs['device']).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in tqdm(range(n_epochs)):
        model.train()
        (hidden, cell) = model.init_state(sequence_length)
        hidden = hidden.to(device)
        cell = cell.to(device)
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred, (hidden, cell) = model(X_batch, (hidden, cell))
            B, T, C = y_pred.shape
            loss = loss_fn(y_pred.view(B * T, C), y_batch.view(B * T))
            hidden = hidden.detach()
            cell = cell.detach()
            loss.backward()
            optimizer.step()

        model.eval()
        loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                y_pred, (hidden, cell) = model(X_batch, (hidden, cell))
                B, T, C = y_pred.shape
                loss = loss_fn(y_pred.view(B * T, C), y_batch.view(B * T))
        
        if epoch % 10 == 0:
            if loss < min_loss:
                save_model(epoch, model, optimizer, loss)
                min_loss = loss

def save_model(epoch, model, optimizer, loss):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, "./cache/model.pth")
    return