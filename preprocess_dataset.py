from datasets import load_dataset
from cleaning_utils import clean_numbers, clean_text
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
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
        y = torch.tensor(y.astype(int).to_numpy(), dtype=torch.long)
        data[split] = [split_tensor, y]

    return data

if __name__ == '__main__':
    data = process_scicite('./data')
    for split, [X, y] in data.items():
        print(split, X.shape, y.shape, torch.unique(y))