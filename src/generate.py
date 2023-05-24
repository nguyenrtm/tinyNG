import torch
from .model import Model
from .preprocessing import create_vocab

def generate_by_names(path, input, max_new_names, **kwargs):
    model = Model(vocab_size=kwargs['vocab_size'],
              embedding_dim=kwargs['embedding_dim'],
              hidden_size=kwargs['hidden_size'],
              dropout=kwargs['dropout'],
              num_layers=kwargs['num_layers'],
              vocab=create_vocab(kwargs['alphabet']),
              device=kwargs['device']).to(kwargs['device'])
    
    model = load_model(path, model, kwargs['device'])
    return model.generate_by_names(input, max_new_names)

def generate_with_letter(path, input, max_new_names, **kwargs):
    model = Model(vocab_size=kwargs['vocab_size'],
              embedding_dim=kwargs['embedding_dim'],
              hidden_size=kwargs['hidden_size'],
              dropout=kwargs['dropout'],
              num_layers=kwargs['num_layers'],
              vocab=create_vocab(kwargs['alphabet']),
              device=kwargs['device']).to(kwargs['device'])
    
    model = load_model(path, model, kwargs['device'])
    return model.generate_with_letter(input, max_new_names)

def load_model(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model