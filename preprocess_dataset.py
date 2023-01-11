from datasets import load_dataset
from cleaning_utils import clean_numbers, clean_text
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from transformers import BertModel, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import contractions
import torch

def process_scicite_glove(X, max_length=60, emb_dim=300):
    tokenizer = get_tokenizer('basic_english')
    embedding = GloVe('6B', dim=emb_dim)
    split_tensor = torch.zeros((X.shape[0], max_length, emb_dim), dtype=torch.float32)
    for i, X_row in tqdm(enumerate(X)):
        emb = embedding.get_vecs_by_tokens(tokenizer(X_row))
        select_size = min(emb.shape[0], max_length)
        split_tensor[i, :select_size, :] = emb[:select_size, :].clone()
    
    return split_tensor

def process_scicite_scibert(X, max_length=60):
    emb_dim = 768
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
    split_tensor = torch.zeros((X.shape[0], max_length, emb_dim), dtype=torch.float32)
    with torch.no_grad():
        for i, X_row in tqdm(enumerate(X)):
            input_ids = torch.tensor([tokenizer.encode(X_row, add_special_tokens=True)[:512]])
            hidden_st = model(input_ids)[0].numpy().reshape(-1, emb_dim)
            select_size = min(hidden_st.shape[0], max_length)
            split_tensor[i, :select_size, :] = torch.from_numpy(hidden_st[:select_size, :])

    return split_tensor

def process_data(data_path, emb_type='glove', emb_size=300, max_length=60):
    data = {}
    dataset = load_dataset('scicite', cache_dir=data_path)
    for split, dataset_split in tqdm(dataset.items()):
        split_df = dataset_split.to_pandas()
        split_df['string'] = split_df['string'].apply(lambda context: context.lower())
        split_df['string'] = split_df['string'].apply(contractions.fix)
        split_df['string'] = split_df['string'].apply(clean_numbers)
        split_df['string'] = split_df['string'].apply(clean_text)
        split_df = split_df.astype(object).replace(np.nan, 'None')
        X, y = split_df['string'], split_df['label']
        X_tensor = None

        if emb_type == 'scibert':
            X_tensor = process_scicite_scibert(X, max_length=max_length)
        elif emb_type == 'glove':
            X_tensor = process_scicite_glove(X, max_length=max_length, emb_dim=emb_size)
        else:
            raise NotImplementedError

        y = torch.tensor(y.astype(int).to_numpy(), dtype=torch.long)
        torch.save({
            'X': X_tensor,
            'y': y
        }, '{}_{}.pth'.format(emb_type, split))
        data[split] = [X_tensor, y]

    return data

def get_data_loaders(emb_type, batch_size):
    train_data = torch.load('{}_train.pth'.format(emb_type))
    val_data = torch.load('{}_validation.pth'.format(emb_type))
    test_data = torch.load('{}_test.pth'.format(emb_type))
    train_dataset = TensorDataset(train_data['X'], train_data['y'])
    val_dataset = TensorDataset(val_data['X'], val_data['y'])
    test_dataset = TensorDataset(test_data['X'], test_data['y'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
        
if __name__ == '__main__':
    process_data('./data', emb_type='glove')