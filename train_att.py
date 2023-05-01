from dataset.dataset_att import Sentiment_Dataset, Senti_Prompt_Data, get_data_loader
# from dataset.dataset_keyword import CommonGenDataset as kw_CommonGenDataset

from dataset.wiki_dataset import WikiDataset, WikiDataset_General, get_wiki_data_loader
# from dataset.e2e_dataset import Seq2SeqDataset, seq_get_data_loader


import argparse
import torch
import torch.nn as nn
import numpy as np
from transformers import T5Tokenizer
import utils
from tqdm import tqdm
import math
import os, sys
from speaksee import evaluation
import spacy
import random
import json
import string
import numpy as np
from os.path import join
import datetime 



from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertTokenizer,
    GPT2Tokenizer 
)

from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM, T5ForConditionalGeneration



import numpy as np
import torch, math, time, os, argparse, re
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
# from adaVAE import compute_loss
from utils import *
from collections import defaultdict
import datetime

import copy as _copy

from torch.utils.data import Dataset, DataLoader
from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D


import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertTokenizer,
    GPT2Tokenizer
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM

from adapters.attribute_distill import Distill_Tuning as Prompt_Residual_Tuning

from adapters.distill_tuning_vanilla import GPT2_Tuning as Vanilla_Prompt_Tuning
from adapters.distill_tuning import Distill_Tuning as Residual_Tuning


from eval_metric import * 
from utils import addCsv


def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--model_name_or_path", type=str, default='/home/zhanghanqing/pretrained_model/gpt2/large')
    parser.add_argument("--steer_model", type=str, default='/home/zhanghanqing/pretrained_model/gpt2/small')
    parser.add_argument("--data_path", type=str, default='../data/pos_neg')
    
    parser.add_argument("--embedding_checkpoint", type=str, default=None)
    parser.add_argument("--task_name", type=str, default="sentiment",choices = ["detoxic","sentiment"])

    parser.add_argument("--pseudo_token", type=str, default='xxx')
    
    parser.add_argument("--batch_size", type=int, default= 100)
    parser.add_argument("--epoch", type=int, default= 50)

    parser.add_argument("--template", type=str, default="(20, 20)")
    parser.add_argument("--early_stop", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # lama configuration
    parser.add_argument("--only_evaluate", type=bool, default=False)
    parser.add_argument("--use_original_template", type=bool, default=False)

    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # directories
    parser.add_argument("--out_dir", type=str, default= './checkpoint')
    # MegatronLM 11B
    
    ## generation configure
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--generated_len", type=int, default=20)
    
    parser.add_argument("--ranking_scope", type=int, default=50)
    
    
    
    parser.add_argument("--corpus_type", type=str, default="positive")
    parser.add_argument("--disc_embedding_checkpoint", type=str, default= None)
    parser.add_argument("--template_disc", type=str, default="(2, 3)")

    parser.add_argument("--max_prompt_length", type=int, default=10)
    parser.add_argument("--training_sample_num", type=int, default=100)
    
    
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--step_log", type=int, default=10000)
    parser.add_argument("--num_layer", type=int, default=2) 
    parser.add_argument("--residual_layer", type=int, default=4)

    
    # parser.add_argument("--pattern", type=str, default="vanilla", choices=["dynamic_prompt_max","dynamic_prompt_mean","dynamic_prompt_hybird","vanilla"])
    parser.add_argument("--tuning_mode", type=str, default="pt", choices=["fp","pt"])
    parser.add_argument("--pretrain_plm", type=str, default="gpt", choices=["gpt","t5"])
    parser.add_argument("--train_stage", type=str, default="fine_tuning", choices=["fine_tuning","general_pretrain","control_pretrain"])
    parser.add_argument("--model_type", type=str, default="Vanilla_Prompt_Tuning", choices=["Residual_Tuning","Prompt_Residual_Tuning","Vanilla_Prompt_Tuning"])
    parser.add_argument("--dataset", type=str, default="CommonGen", choices=["CommonGen","keyword","roc"])


    # parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--memory_p", type=float, default=0.5)

    
    parser.add_argument("--output_path", type=str, default="../eval")
    parser.add_argument("--mode", type=str, default="ctg", choices=["ctg","train","classifer"])
    parser.add_argument("--evaluate_file", type=str, default="../our_text")
    parser.add_argument("--evaluate_outfile", type=str, default="./eval/our/result.csv")
    parser.add_argument("--max_epoch", type=int, default=10)
    
   
    parser.add_argument("--check_point_load", type=str, default= None)
    parser.add_argument("--copy_vocab_path", type=str, default= None)
    parser.add_argument("--train_path", type=str, default= None)
    parser.add_argument("--dev_path", type=str, default= None)
    parser.add_argument("--test_path", type=str, default= None)
    parser.add_argument("--pretrain_path", type=str, default= None)
    parser.add_argument("--pretrain_path_val", type=str, default= None)
    parser.add_argument("--long_test_path", type=str, default= None)

    
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--test', action='store_true')
    
    parser.add_argument('--saving_model', action='store_true')

    
    args = parser.parse_args()
    # post-parsing args
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template
    # args.template_disc = eval(args.template_disc) if type(args.template_disc) is not tuple else args.template_disc

    assert type(args.template) is tuple

    return args





def run_eval(args, model, eval_data_iter, tokenizer, output_path=None):
    model.eval()

    gts = []
    concept_set = []
    
    res = []
    gens_part = []
    context_part = []
    
    with torch.no_grad():
        for batch in tqdm(eval_data_iter):
            
            input_ids =  batch["input_ids"].to(args.device).long()
            
            encode_inputs =  batch["encode_input"].to(args.device).long() 
            attention_mask = batch["attention_mask"].to(args.device).bool()
                        

            output_sequences = model.generate(
                input_ids=input_ids,
                encoder_hidden_states = encode_inputs,
                attention_mask = attention_mask,
                max_length =args.max_length + input_ids.shape[1],
                num_beams =1,
                top_p = 0.5,
                repetition_penalty=1.25,
                top_k = 0,
                no_repeat_ngram_size = 3,
                do_sample= True, # disable sampling to test if batching affects output
            )
            text = []
            context_text = [] 
            generated_text = []
            for i in range(len(output_sequences)):
                text.append(tokenizer.decode(output_sequences[i],skip_special_tokens= True))
                context_text.append(tokenizer.decode(input_ids[i],skip_special_tokens= True))
                generated_text.append(tokenizer.decode(output_sequences[i][input_ids.shape[1]:],skip_special_tokens= True))
                

            res += text
            
            context_text = [t.strip() for t in context_text]            
            context_part += context_text
            
            generated_text = [t.strip() for t in generated_text]            
            gens_part += generated_text
            
            print(text)
            
            
#     if args.validation or args.test:
        
#         for key, c,g,r in zip(concept_set, context_part, gens_part, res):
#             dict_data = {
#                 "keywords":key,
#                 "context":c,
#                 "generated":g,
#                 "text":r}
#             addCsv(output_path+f"/generated_result_{args.generated_len}_seed_{args.seed}.csv", dict_data)
#         print("The result is generated!")
    





def task_train(args, model, tokenizer, train_data_loader, dev_data_loader, test_data_loader, optimizer, my_lr_scheduler):
    
    result_name_path = f"{args.output_path}/{args.model_type}_seed_{args.seed}_{args.tuning_mode}_training_samples_{args.training_sample_num}_memory_p_{args.memory_p}_layer_{args.residual_layer}.csv"
        
    best_score = 0.0   
    early_stop=0
    coverage = 0.0
    time_record = str(datetime.datetime.now()).replace(" ","_")


    if args.train:
        print("len(max_epoch):", args.max_epoch)
        for epoch in range(args.max_epoch):
            print('EPOCH %d / %d' % (epoch + 1, args.max_epoch))
            tot_loss = 0
            model.train()
            step = 0 
            step_count=0
            
            for batch_idx, batch in tqdm(enumerate(train_data_loader)):
                    model.train()
                                    
                    input_ids =  batch["input_ids"].to(args.device).long()
                    encode_inputs =  batch["encode_input"].to(args.device).long()
                    encode_inputs_ =  batch["encode_input_"].to(args.device).long()
                    attention_mask =  batch["attention_mask"].to(args.device).bool()

                    output =  model(encoder_hidden_states=encode_inputs, labels=encode_inputs_, input_ids = input_ids, attention_mask=attention_mask)
                    
                    loss = output.loss
                    print("the loss is:", loss)
                    tot_loss += loss.item()

                    loss.backward()
                    torch.cuda.empty_cache()
                    optimizer.step()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
            
                    step += args.batch_size
                    step_count += args.batch_size
                    
            my_lr_scheduler.step()

            if args.saving_model:
                save_model(args, model, args.seed, time_record+"_"+str(args.memory_p))    





if __name__ == "__main__":

    args = construct_generation_args()
    
    print("Whether Saving_model:", args.saving_model)
    print("Text generation max length:",args.max_length)
    
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)#as reproducibility docs
    torch.manual_seed(seed)# as reproducibility docs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False# as reproducibility docs
    torch.backends.cudnn.deterministic = True# as reproducibility docs
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
        
    model = Prompt_Residual_Tuning.from_pretrained(args.model_name_or_path)
    model.init_post(args)    

        
    
    if args.check_point_load != None and  hasattr(model, 'prompt_encoder'):
        model.prompt_encoder.load_state_dict(load_prompt(args.check_point_load))
        print("load the embedding checkpoint successfully!")
        
    model.to(args.device)
    

    print("args.batch_size:",args.batch_size)
    
    if args.validation or args.test:
        
        test_data = Senti_Prompt_Data(args.test_path, tokenizer, is_training=False, args=args)
        
        test_data_loader = get_data_loader(test_data, args.batch_size)
        
        run_eval(args, model, test_data_loader, tokenizer, output_path=args.output_path)
  
        exit()
        
        
    
    
    if args.train:

        params = [{'params': model.prompt_encoder.parameters()}]
        optimizer = torch.optim.AdamW(params,  weight_decay= args.weight_decay,lr=args.lr)
        
        if args.train_stage == "fine_tuning":
            my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size= 3, gamma=0.2)
        else:
            my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.5)
            
            
        train_data = Sentiment_Dataset(args.train_path, tokenizer, is_training=True, args=args)
        print("train_data:", len(train_data))
        train_data_loader = get_data_loader(train_data, args.batch_size)
            

        task_train(args, model, tokenizer, train_data_loader, None, None, optimizer, my_lr_scheduler)
            


        

        
    else:
         raise Exception("the task is out of scope!")
        
    

    
    

        
       
