import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def reader(path):
    with open(path) as file:
        text = file.read()
    return text

def create_vocab(alphabet):
    chars = sorted(list(set(alphabet)))
    vocab = { ch:i for i,ch in enumerate(chars) }
    return vocab

def get_key(val, vocab):
    for key, value in vocab.items():
        if val == value:
            return key
 
    return f"key {val} doesn't exist"

def txt_to_tensor(txt, vocab, device):
    result_array = [vocab[c] for c in txt]
    return torch.tensor(result_array, dtype=int).to(device)

def tensor_to_txt(tensor, vocab):
    result = ''.join([get_key(c, vocab) for c in tensor])
    return result

def divide_data(data, seq_length, device):
    seq_length += 1
    result = torch.empty((0, seq_length), dtype=int).to(device)
    n = int(len(data) / seq_length)
    for i in tqdm(range(n)):
        tmp = data[i * seq_length:(i + 1) * seq_length]
        tmp = tmp.reshape([1, seq_length])
        result = torch.cat((result, tmp), 0)
    return result

def split_input_target(tensor, seq_length, device):
    return (tensor[:-1].reshape(1, seq_length).to(device), tensor[1:].reshape(1, seq_length).to(device))

def get_input_target_dataset(data, seq_length, device):
    X = torch.empty((0, seq_length), dtype=int).to(device)
    y = torch.empty((0, seq_length), dtype=int).to(device)
    len = data.size()[0]
    for i in tqdm(range(len)):
        (X_tmp, y_tmp) = split_input_target(data[i], seq_length, device)
        X = torch.cat((X, X_tmp), 0)
        y = torch.cat((y, y_tmp), 0)
    return (X, y)

def preprocessing(path, **kwargs):
    device = kwargs['device']
    seq_length = int(kwargs['seq_length'])
    alphabet = kwargs['alphabet']
    text = reader(path)
    vocab = create_vocab(alphabet)
    data_train = txt_to_tensor(text, vocab, device)
    divided_data = divide_data(data_train, seq_length, device)
    (X, y) = get_input_target_dataset(divided_data, seq_length, device)
    X = X.to(device)
    y = y.to(device)
    loader = DataLoader(list(zip(X,y)), shuffle=True, batch_size=64)
    return loader