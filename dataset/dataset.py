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
                        "item": 0})
                        
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
        self.start_prompt = "$generation$"
        self.args = args

        
        self.tokenizer.padding_side = "right"
        self.read_content(json_path)
        # self.tokenizer.padding_side = "right"
        
        
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
                self.start_prompt = concept_set

                if self.args.pretrain_plm == "gpt":
                    # concept_set_input_ids = self.tokenizer("<|endoftext|> " + concept_set, return_tensors="np")['input_ids'][0][1:].tolist()
                    concept_set_input_ids = self.tokenizer(concept_set, return_tensors="np")['input_ids'][0].tolist()

                else:
                    concept_set_input_ids = self.tokenizer(concept_set, return_tensors="np")['input_ids'][0].tolist()

                gt = copy.deepcopy(item['scene'])
 
                if self.is_training:
                    for c in item['scene']:
                        c = c.lower().replace('\n', '')
                        if c.endswith('.'):
                            c= c[:-1]                    
  
                        if self.args.pretrain_plm == "gpt":
                            c_output_ids = self.tokenizer(c, return_tensors="np")['input_ids'][0].tolist()                            
                        else: 
                            c_output_ids = self.tokenizer(c, return_tensors="np")['input_ids'][0].tolist()
                        
                                    
                        self.record.append({
                            "concept_set":item['concept_set'],
                            "concept_set_input_ids":concept_set_input_ids,
                            "c_output_ids":c_output_ids,
                            "item": 0
                        })
                else:
                    self.record.append({
                            "concept_set":item['concept_set'],
                            "concept_set_input_ids":concept_set_input_ids,
                            "c_output_ids":np.array([0,0]),
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
                  'item':[d['item'] for d in dataset]}

    _PAD = tokenizer.eos_token_id
    _EOS = tokenizer.eos_token_id
    
    
    max_concept_len = max([len(d['concept_set_input_ids']) for d in dataset])

    
    concept_set_input = np.full((batch_size, max_concept_len), _PAD, dtype=np.int64)
    
    for i, d in enumerate(dataset):
        # concept_set_input[i, :len(d['concept_set_input_ids'])] = d['concept_set_input_ids']
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



def get_data_loader(dataset, batch_size):
    collate_fn = lambda d: data_wrapper(d, dataset.tokenizer, dataset.args.pretrain_plm)
    return DataLoader(dataset, 
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )
