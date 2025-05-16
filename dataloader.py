import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import os

class TransliterationDataset(Dataset):
    def __init__(self, file_path):
        self.source_texts = []
        self.target_texts = []
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split('\t')
                
                # Handle different file formats
                if len(parts) == 3:  # Format: Hindi\tRomanized\tFrequency
                    target, source, _ = parts
                    self.source_texts.append(source)
                    self.target_texts.append(target)
                elif len(parts) == 2:  # Format: Hindi\tRomanized
                    target, source = parts
                    self.source_texts.append(source)
                    self.target_texts.append(target)
                else:
                    print(f"Warning: Line {line_num} has unexpected format (expected 2 or 3 columns): {line.strip()}")
        
        # If no data was loaded, raise an error
        if len(self.source_texts) == 0:
            raise ValueError(f"No data was loaded from {file_path}. File may be empty or incorrectly formatted.")
                    
        self.source_char_set = set()
        self.target_char_set = set()
        
        for source in self.source_texts:
            self.source_char_set.update(source)
        
        for target in self.target_texts:
            self.target_char_set.update(target)
        
        self.source_char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(self.source_char_set))}
        self.source_idx_to_char = {idx + 1: char for idx, char in enumerate(sorted(self.source_char_set))}
        self.source_char_to_idx['<PAD>'] = 0
        self.source_idx_to_char[0] = '<PAD>'
        self.source_char_to_idx['< SOS >'] = len(self.source_char_to_idx)
        self.source_idx_to_char[len(self.source_idx_to_char)] = '< SOS >'
        self.source_char_to_idx['<EOS>'] = len(self.source_char_to_idx)
        self.source_idx_to_char[len(self.source_idx_to_char)] = '<EOS>'
        
        self.target_char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(self.target_char_set))}
        self.target_idx_to_char = {idx + 1: char for idx, char in enumerate(sorted(self.target_char_set))}
        self.target_char_to_idx['<PAD>'] = 0
        self.target_idx_to_char[0] = '<PAD>'
        self.target_char_to_idx['< SOS >'] = len(self.target_char_to_idx)
        self.target_idx_to_char[len(self.target_idx_to_char)] = '< SOS >'
        self.target_char_to_idx['<EOS>'] = len(self.target_char_to_idx)
        self.target_idx_to_char[len(self.target_idx_to_char)] = '<EOS>'
        
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        source = self.source_texts[idx]
        target = self.target_texts[idx]
        
        source_indices = [self.source_char_to_idx['< SOS >']] + [self.source_char_to_idx[char] for char in source] + [self.source_char_to_idx['<EOS>']]
        target_indices = [self.target_char_to_idx['< SOS >']] + [self.target_char_to_idx[char] for char in target] + [self.target_char_to_idx['<EOS>']]
        
        return {
            'source': source,
            'target': target,
            'source_indices': source_indices,
            'target_indices': target_indices,
            'source_length': len(source_indices),
            'target_length': len(target_indices)
        }
    
    @property
    def source_vocab_size(self):
        return len(self.source_char_to_idx)
    
    @property
    def target_vocab_size(self):
        return len(self.target_char_to_idx)

def collate_fn(batch):
    max_source_len = max([len(item['source_indices']) for item in batch])
    max_target_len = max([len(item['target_indices']) for item in batch])
    
    source_batch = []
    target_batch = []
    source_lengths = []
    target_lengths = []
    
    source_texts = []
    target_texts = []
    
    for item in batch:
        source_padded = item['source_indices'] + [0] * (max_source_len - len(item['source_indices']))
        target_padded = item['target_indices'] + [0] * (max_target_len - len(item['target_indices']))
        
        source_batch.append(source_padded)
        target_batch.append(target_padded)
        source_lengths.append(item['source_length'])
        target_lengths.append(item['target_length'])
        
        source_texts.append(item['source'])
        target_texts.append(item['target'])
    
    return {
        'source': torch.LongTensor(source_batch),
        'target': torch.LongTensor(target_batch),
        'source_lengths': torch.LongTensor(source_lengths),
        'target_lengths': torch.LongTensor(target_lengths),
        'source_texts': source_texts,
        'target_texts': target_texts
    }

def get_dataloader(file_path, batch_size, shuffle=True):
    try:
        dataset = TransliterationDataset(file_path)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )
        return dataloader, dataset
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print(f"Please make sure the Dakshina dataset is downloaded and located correctly.")
        print(f"Expected file path: {file_path}")
        print(f"Current working directory: {os.getcwd()}")
        raise
    except ValueError as e:
        print(f"Error: {str(e)}")
        raise