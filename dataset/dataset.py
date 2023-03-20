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


class C2Gen(Dataset):
    def __init__(self, json_path, tokenizer, args=None):
        super(C2Gen, self).__init__()

        self.tokenizer = tokenizer
        np.set_printoptions(threshold=sys.maxsize)
        self.args = args
        
        self.record = []
        self.read_content(json_path)
        
    def read_content(self, json_path):
        print("reading data from %s ..." % json_path)
        
        data = json.load(open(json_path,"r"))
        for r in data["rows"]:            
            keyword = ' '.join(r["row"]["keywords"])
            context =  r["row"]["context"]
            concept = '#'.join(r["row"]["keywords"])
            keyword = self.tokenizer(keyword, return_tensors="np")['input_ids'][0].tolist()
            context = self.tokenizer(context, return_tensors="np")['input_ids'][0].tolist()

            
            self.record.append({
                        "concept_set_input_ids":keyword,
                        "c_output_ids":context,
                        "concept_set":concept,
                        "item": 0,
                        "cat_text":[0]})
                        
        random.shuffle(self.record)
          

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]
        return item


class CommonGenDataset(Dataset):

    def __init__(self, json_path, tokenizer, is_training=False, args=None):
        super(CommonGenDataset, self).__init__()

        self.tokenizer = tokenizer
        self.is_training = is_training
        np.set_printoptions(threshold=sys.maxsize)
        self.args = args

        
        self.tokenizer.padding_side = "right"
        self.read_content(json_path)
        
        # self.generation_templete = "Include: with "
        
        
    def read_content(self, json_path):
        print("reading data from %s ..." % json_path)
        self.record = []
        with open(json_path) as out:
            if self.is_training:
                # lines = out.readlines()[:100]
                random.seed(42)
                lines = random.sample(out.readlines(), self.args.training_sample_num)
            else:
                lines = out.readlines()
                
            for l in tqdm(lines):
                item = json.loads(l.strip())
                concept_set = ' '.join(item['concept_set'].split('#'))

                concept_set_input_ids = self.tokenizer(f"Includes: {concept_set} # ", return_tensors="np")['input_ids'][0].tolist()


                gt = copy.deepcopy(item['scene'])
 
                if self.is_training:
                    for c in item['scene']:
                        c = c.strip()
                        if c.endswith('.'):
                            c= c[:-1].strip()                   
  
                        c_output_ids = self.tokenizer(c, return_tensors="np")['input_ids'][0].tolist()

                        self.record.append({
                            "concept_set":item['concept_set'],
                            "concept_set_input_ids":concept_set_input_ids,
                            "c_output_ids":c_output_ids,
                            "cat_text": concept_set_input_ids+c_output_ids,
                            "item":0})
                else:
                    self.record.append({
                            "concept_set":item['concept_set'],
                            "concept_set_input_ids":concept_set_input_ids,
                            "c_output_ids":np.array([0,0]),
                            "cat_text": concept_set_input_ids,
                            "item": item['scene']})
                        
        if self.is_training: random.shuffle(self.record)

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]

        return item


def data_wrapper(dataset, tokenizer, plm_type):
    batch_size = len(dataset)
    new_dataset = {'concept_set': [d['concept_set'] for d in dataset], 
                   'concept_set_input_ids': [d['concept_set_input_ids'] for d in dataset],
                   'c_output_ids': [d['c_output_ids'] for d in dataset],                   
                   'cat_text':[d['cat_text'] for d in dataset],
                  'item':[d['item'] for d in dataset]}

    _PAD = tokenizer.eos_token_id
    _EOS = tokenizer.eos_token_id
    
    
    max_concept_len = max([len(d['concept_set_input_ids']) for d in dataset])
    concept_set_input = np.full((batch_size, max_concept_len), _PAD, dtype=np.int64)
    for i, d in enumerate(dataset):
        data = d['concept_set_input_ids'][:max_concept_len]
        concept_set_input[i, -len(data):] = data
    new_dataset['concept_set_input_ids'] = torch.from_numpy(concept_set_input)
    
    
                           
    max_output_len = max([len(d['c_output_ids']) for d in dataset])
    output_ids = np.full((batch_size, max_output_len), _PAD, dtype=np.int64)
    for i, d in enumerate(dataset):
        output_ids[i, :len(d['c_output_ids'])] = d['c_output_ids']
    new_dataset['c_output_ids'] = torch.from_numpy(output_ids)
    
    
    max_output_len = max([len(d['cat_text']) for d in dataset])
    input_ids = np.full((batch_size, max_output_len), _PAD, dtype=np.int64)
    for i, d in enumerate(dataset):
        input_ids[i, :len(d['cat_text'])] = d['cat_text']
    new_dataset['cat_text'] = torch.from_numpy(input_ids)
    
    
    max_output_len = max([len(d['cat_text']) for d in dataset])
    mask_ids = np.full((batch_size, max_output_len), 1, dtype=np.int64)
    for i, d in enumerate(dataset):
        mask_ids[i, :len(d['concept_set_input_ids'])] = 0
    new_dataset['mask_ids'] = torch.from_numpy(mask_ids)
        
    
                          
    return new_dataset



def get_data_loader(dataset, batch_size):
    collate_fn = lambda d: data_wrapper(d, dataset.tokenizer, dataset.args.pretrain_plm)
    return DataLoader(dataset, 
        batch_size=batch_size,
        num_workers=2,
        collate_fn=collate_fn
    )
