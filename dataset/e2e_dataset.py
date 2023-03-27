from torch.utils.data import Dataset, DataLoader
import linecache
import json
import random
from tqdm import tqdm
import numpy as np
import torch
import copy
import re
import string
import sys
import random
from pathlib import Path


class Seq2SeqDataset(Dataset):
    def __init__(
        self, json_path, tokenizer, type_path="train", args=None):
        super().__init__()
        
        self.src_file = Path(json_path).joinpath(type_path + ".src")
        self.tgt_file = Path(json_path).joinpath(type_path + ".tgt")
        self.tokenizer = tokenizer
        
        self.data_lens = len(Path(self.src_file ).open().readlines())
        # self.data_lens = 100

        self.record = []
        self.read_content()
        
        
    def read_content(self):
        
        for i in range(self.data_lens):
            index = i+1
            src_line = linecache.getline(str(self.src_file), index).rstrip("\n")          
            tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
            
            input_ids = self.tokenizer(src_line, return_tensors="np")['input_ids'][0].tolist()
            output_ids = self.tokenizer(tgt_line, return_tensors="np")['input_ids'][0].tolist()
            
            self.record.append({
                            "input_ids":input_ids,
                            "output_ids":output_ids})        
        
    def __len__(self):
        return  self.data_lens

    def __getitem__(self, index):
        item = self.record[index]

        return item
        


def data_wrapper(dataset, tokenizer):
    batch_size = len(dataset)
    new_dataset = {'input_ids': [d['input_ids'] for d in dataset], 
                   'output_ids': [d['output_ids'] for d in dataset]}

    _PAD = tokenizer.eos_token_id
    
    max_concept_len = max([len(d['input_ids']) for d in dataset])
    input_data = np.full((batch_size, max_concept_len), _PAD, dtype=np.int64)
    
    for i, d in enumerate(dataset):
        data = d['input_ids'][:max_concept_len]
        input_data[i, :len(data)] = data
    new_dataset['input_ids'] = torch.from_numpy(input_data)
    
    
                           
    max_output_len = max([len(d['output_ids']) for d in dataset])
    output_ids = np.full((batch_size, max_output_len), _PAD, dtype=np.int64)
    
    for i, d in enumerate(dataset):
        output_ids[i, :len(d['output_ids'])] = d['output_ids']
    new_dataset['output_ids'] = torch.from_numpy(output_ids)
                          
    return new_dataset



def seq_get_data_loader(dataset, batch_size):
    collate_fn = lambda d: data_wrapper(d, dataset.tokenizer)
    return DataLoader(dataset, 
        batch_size=batch_size,
        num_workers=2,
        collate_fn=collate_fn
    )