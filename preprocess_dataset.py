from datasets import load_dataset
from cleaning_utils import clean_numbers, clean_text
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import contractions
import torch

def process_scicite(data_path, max_length=60, emb_dim=300):
    data = {}
    dataset = load_dataset('scicite', cache_dir=data_path)
    tokenizer = get_tokenizer('basic_english')
    embedding = GloVe('6B', dim=emb_dim)
    for split, dataset_split in tqdm(dataset.items()):
        split_df = dataset_split.to_pandas()
        split_tensor = torch.zeros((split_df.shape[0], max_length, emb_dim), dtype=torch.float32)
        split_df['string'] = split_df['string'].apply(lambda context: context.lower())
        split_df['string'] = split_df['string'].apply(contractions.fix)
        split_df['string'] = split_df['string'].apply(clean_numbers)
        split_df['string'] = split_df['string'].apply(clean_text)
        split_df = split_df.astype(object).replace(np.nan, 'None')
        X, y = split_df['string'], split_df['label']
        for i, (X_row, y_row) in tqdm(enumerate(zip(X, y))):
            emb = embedding.get_vecs_by_tokens(tokenizer(X_row))
            select_size = min(emb.shape[0], max_length)
            split_tensor[i, :select_size, :] = emb[:select_size, :].clone()
        torch.save({
            'X': split_tensor,
            'y': y
        }, 'glove_{}.pth'.format(split))
        y = torch.tensor(y.astype(int).to_numpy(), dtype=torch.long)
        data[split] = [split_tensor, y]

    return data

def process_scicite_scibert(data_path):
    data = {}
    emb_dim = 768
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
    dataset = load_dataset('scicite', cache_dir=data_path)
    for split, dataset_split in tqdm(dataset.items()):
        split_df = dataset_split.to_pandas()
        split_tensor = torch.zeros((split_df.shape[0], emb_dim), dtype=torch.float32)
        split_df['string'] = split_df['string'].apply(lambda context: context.lower())
        split_df['string'] = split_df['string'].apply(contractions.fix)
        split_df['string'] = split_df['string'].apply(clean_numbers)
        split_df['string'] = split_df['string'].apply(clean_text)
        split_df = split_df.astype(object).replace(np.nan, 'None')
        X, y = split_df['string'], split_df['label']
        with torch.no_grad():
            for i, (X_row, y_row) in tqdm(enumerate(zip(X, y))):
                input_ids = torch.tensor([tokenizer.encode(X_row, add_special_tokens=True)[:512]])
                hidden_st = model(input_ids)[1].numpy()
                split_tensor[i, :] = torch.from_numpy(hidden_st[0, :])
        y = torch.tensor(y.astype(int).to_numpy(), dtype=torch.long)
        torch.save({
            'X': split_tensor,
            'y': y
        }, 'scibert_{}.pth'.format(split))
        data[split] = [split_tensor, y]

    return data

if __name__ == '__main__':
    data = process_scicite_scibert('./data')
    for split, [X, y] in data.items():
        print(split, X.shape, y.shape, torch.unique(y))