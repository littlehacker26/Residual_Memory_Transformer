from torch.utils.data import Dataset, DataLoader
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
from glob import glob
from pathlib import Path
import os
from multiprocessing import Pool
from copy import deepcopy




class WikiDataset_General(Dataset):

    def __init__(self, json_path, tokenizer, is_training=False, args=None):
        super(WikiDataset_General, self).__init__()

        self.tokenizer = tokenizer
        self.is_training = is_training
        np.set_printoptions(threshold=sys.maxsize)
        self.args = args
        
        self.record = []
        
        self.mask_token_id = tokenizer.encode("<|endoftext|> "+ args.pseudo_token)[-1]

        self.read_content(json_path)
        
        
    def read_content(self, json_path):
        print("reading data from %s ..." % json_path)
        
        filenames = []
        filenames += glob(os.path.join(json_path,'wiki**.format'))
                
        data =  []
        
        for data_file in filenames:
            data +=[x for x in Path(data_file).open().readlines()[:self.args.training_sample_num]]
            
        n = int(len(data)/128)
        data_list =[data[i:i + n] for i in range(0, len(data), n)]
        
        with Pool(128) as p:
            
            data =list(tqdm(p.imap(self.encoder_data, data_list), total=len(data_list)))
            
        self.record = [item for subl in data for item in subl]

                        
        if self.is_training: random.shuffle(self.record)
        
    
    def encoder_data(self, data):
        
        res = []
        for d in data:
            concept_set_input_ids = self.tokenizer(d, return_tensors="np")['input_ids'][0].tolist()

            if len(concept_set_input_ids)<3 or len(concept_set_input_ids)>=150:
                continue
                
            length = len(concept_set_input_ids)
            sentinal_count = max(int(length * 0.2), 1)
            
            sentinal_positions = random.sample(range(1, length-1), sentinal_count)
            _concept_set_input_ids = deepcopy(concept_set_input_ids)
            
                
            if sentinal_positions[0]%6<3:# replace span on the prob of 1/2
                for i in sentinal_positions:
                    _concept_set_input_ids[i] = self.mask_token_id
                
                new_a = []
                flag=0
                for i in _concept_set_input_ids:
                    if i == self.mask_token_id:
                        flag+=1
                        if flag==1:
                            new_a.append(i)
                    else:
                        flag=0
                        new_a.append(i)
                _concept_set_input_ids = new_a
                    
            elif sentinal_positions[0]%6==3:# Document rotation on the prob of 1/6
                _concept_set_input_ids = _concept_set_input_ids[sentinal_positions[0]:]+_concept_set_input_ids[: sentinal_positions[0]]
                                
                
            elif sentinal_positions[0]%6==4: # delete operation on the prob of 1/6
                for i in sentinal_positions:
                    _concept_set_input_ids[i] = self.mask_token_id
                _concept_set_input_ids= list(filter(lambda x: x!=self.mask_token_id, _concept_set_input_ids))
                
            elif sentinal_positions[0]%6==5: # inseration mask token on the prob of 1/6
                sentinal_positions.sort()
                for index, pos in enumerate(sentinal_positions):
                    _concept_set_input_ids.insert(pos+index, self.mask_token_id)
                
            res.append({
                        "concept_set_input_ids":_concept_set_input_ids,
                        "c_output_ids":concept_set_input_ids})
        return res            
                    

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]
        return item
    



class WikiDataset(Dataset):

    def __init__(self, json_path, tokenizer, is_training=False, args=None):
        super(WikiDataset, self).__init__()

        self.tokenizer = tokenizer
        self.is_training = is_training
        np.set_printoptions(threshold=sys.maxsize)
        self.args = args
        
        self.record = []
        self.read_content(json_path)
        
        
    def read_content(self, json_path):
        print("reading data from %s ..." % json_path)
        
        filenames = []                
        filenames += glob(os.path.join(json_path,'wiki**.format'))
                
        data =  []
        
        for data_file in filenames:
            data +=[x for x in Path(data_file).open().readlines()[:self.args.training_sample_num]]
            
        n = int(len(data)/48)
        data_list =[data[i:i + n] for i in range(0, len(data), n)]
        
        with Pool(48) as p:
            
            data =list(tqdm(p.imap(self.encoder_data, data_list), total=len(data_list)))
            
        self.record = [item for subl in data for item in subl]

                        
        if self.is_training: random.shuffle(self.record)
        
        
    
    def encoder_data(self, data):
        
        res = []
        for d in data:
            concept_set_input_ids = self.tokenizer(d, return_tensors="np")['input_ids'][0].tolist()

            if len(concept_set_input_ids)<10 or len(concept_set_input_ids)>=128:
                continue
            res.append({
                        "concept_set_input_ids":random.sample(concept_set_input_ids, int(len(concept_set_input_ids)*0.2)),
                        "c_output_ids":concept_set_input_ids})
            
        return res            
                    

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]
        return item
    


def data_wrapper(dataset, tokenizer, plm_type):
    batch_size = len(dataset)
    new_dataset = {'c_output_ids': [d['c_output_ids'] for d in dataset], 
                   'concept_set_input_ids': [d['concept_set_input_ids'] for d in dataset]}

    _PAD = tokenizer.eos_token_id
    
    
    max_concept_len = max([len(d['concept_set_input_ids']) for d in dataset])
    concept_set_input = np.full((batch_size, max_concept_len), _PAD, dtype=np.int64)
    
    for i, d in enumerate(dataset):
        data = d['concept_set_input_ids'][:max_concept_len]
        concept_set_input[i, :len(data)] = data
        
    new_dataset['concept_set_input_ids'] = torch.from_numpy(concept_set_input)
    new_dataset['attention_mask'] = (new_dataset['concept_set_input_ids'] != _PAD)
    
                           
    max_output_len = max([len(d['c_output_ids']) for d in dataset])
    output_ids = np.full((batch_size, max_output_len), _PAD, dtype=np.int64)
    
    
    for i, d in enumerate(dataset):
        output_ids[i, :len(d['c_output_ids'])] = d['c_output_ids']
        
    new_dataset['c_output_ids'] = torch.from_numpy(output_ids)
    new_dataset['output_attention_mask'] = (new_dataset['c_output_ids'] != _PAD)
    
                          
    return new_dataset

def get_wiki_data_loader(dataset, batch_size):
    collate_fn = lambda d: data_wrapper(d, dataset.tokenizer, dataset.args.pretrain_plm)
    return DataLoader(dataset, 
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )
