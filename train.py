from dataset.vocabulary import T5CopyVocabulary
from dataset.dataset import CommonGenDataset, C2Gen, get_data_loader
from dataset.wiki_dataset import WikiDataset, WikiDataset_General, get_wiki_data_loader
import argparse
import torch
import torch.nn as nn
from config import Config
import numpy as np
from transformers import T5Tokenizer
from checkpointing import CheckpointManager
import utils
from tqdm import tqdm
import math
import os, sys
from speaksee import evaluation
import spacy
import random
from constraint import CBSConstraint
from dataset.diversity import distinct_n
import json
import string
import numpy as np
from os.path import join


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
import torch, math, time, os, argparse, copy, re
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
# from adaVAE import compute_loss
from utils import *
from collections import defaultdict
from adapters.common import AdapterConfig
from data import ConditionalGenerationDataset, GenerationDataset
import datetime

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

from adapters.distill_tuning_d import Distill_Tuning as Prompt_Residual_Tuning
from adapters.distill_tuning_vanilla import Distill_Tuning as Vanilla_Prompt_Tuning
from adapters.distill_tuning import Distill_Tuning as Residual_Tuning


from adapters.t5_prompt import T5PromptTuning
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
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--max_prompt_length", type=int, default=10)
    parser.add_argument("--training_sample_num", type=int, default=100)
    
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--num_layer", type=int, default=2)


    # parser.add_argument("--pattern", type=str, default="vanilla", choices=["dynamic_prompt_max","dynamic_prompt_mean","dynamic_prompt_hybird","vanilla"])
    parser.add_argument("--tuning_mode", type=str, default="pt", choices=["fp","pt"])
    parser.add_argument("--pretrain_plm", type=str, default="gpt", choices=["gpt","t5"])
    parser.add_argument("--train_stage", type=str, default="fine_tuning", choices=["fine_tuning","general_pretrain","control_pretrain"])
    parser.add_argument("--model_type", type=str, default="Vanilla_Prompt_Tuning", choices=["Residual_Tuning","Prompt_Residual_Tuning","Vanilla_Prompt_Tuning"])


    parser.add_argument("--prompt_pad_length", type=int, default= 10)
    # parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--ranking_scope", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)

    
    parser.add_argument("--output_path", type=str, default="./eval")
    parser.add_argument("--mode", type=str, default="ctg", choices=["ctg","train","classifer"])
    parser.add_argument("--evaluate_file", type=str, default="../our_text")
    parser.add_argument("--evaluate_outfile", type=str, default="./eval/our/result.csv")
    parser.add_argument("--max_epoch", type=int, default=10)
    
   
    # parser.add_argument("--check_point_save", type=str, default= None)
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
    
    args = parser.parse_args()
    # post-parsing args
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template
    # args.template_disc = eval(args.template_disc) if type(args.template_disc) is not tuple else args.template_disc

    assert type(args.template) is tuple

    return args




def post_proces(text, flag_tokens):
    
    res =[]
    for t, f in zip(text, flag_tokens):
        data = t.replace('\n', '').replace('\xa0', '')
        
        for ii in f.keys():
            if ii in data:
                data = data.replace(ii, f[ii])
        
        res.append(data)
    
    return res


def run_eval_ppl(args, model, eval_data_iter, tokenizer, only_test=False, output_path=None):
    
    model.eval()

    ppls = []
    
    with torch.no_grad():
        for batch in tqdm(eval_data_iter):

            x_token = batch["concept_set_input_ids"].to(args.device).long()
            input_ids =  batch["c_output_ids"].to(args.device).long()
            x_mask =  batch["output_attention_mask"].to(args.device).long()
            
            logits,_  =  model(x_token, input_ids)
            ppl      =   ppl_from_pretrained_model(logits, input_ids, x_mask)
            ppls+=ppl
            
    return np.nanmean(ppls)
            
    
    

def run_eval(args, model, eval_data_iter, tokenizer, only_test=True, output_path=None):
    model.eval()

    gts = []
    concept_set = []
    res = []
    
    with torch.no_grad():
        for batch in tqdm(eval_data_iter):
                 
            x_token = batch["concept_set_input_ids"].to(args.device).long()
            input_ids =  batch["c_output_ids"].to(args.device).long()

            gts += batch["item"]
            concept_set += batch["concept_set"]
            
            if only_test == True:
                input_ids = None
                
            result = model.generate(prompts_ids = x_token, max_length=20, context=input_ids)
            text = tokenizer.batch_decode(result["generated_tokens"], skip_special_tokens= True)
            
            text = [t.strip() for t in text]
            print(text)
            res += text
            
    references={}
    hypothesis  = {}
    
    for g, c, r in zip(gts, concept_set, res):
        references[c] = g
        hypothesis[c] = [r]
        # hypothesis[c] = [g[0]]

    res =  hypothesis
    
    gts = references
    
    print("res:", res)
    
    cov = evaluator_coverage(res)
    
    if  only_test == False:
        return {"l_coverage":cov}
    # print("Coverage score is:", a)
    
    # score = evaluator_cider(gts, res)
    # print("Cider: %0.3f" %score)
    
    # score = evaluator_meteor(gts, res)
    # print("Meteor: %0.3f" %score)
    
    score_bleu = evaluator_bleu(gts, res)
    # print("Bleu:", score_bleu)
    
    score_self_bleu = evaluator_selfbleu(res)
    # print("self bleu:", score_self_bleu)
    
    score_ppl = evaluator_ppl(res, "/home2/zhanghanqing/pretrained_model/gpt2/large")
    # print("PPL: %0.3f" score_ppl)
    
    return {"coverage":cov, "bleu":score_bleu,"self_bleu":score_self_bleu, "ppl": score_ppl}
        




    

def create_model_gpt(model_name_or_path):
    
    if model_name_or_path:
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        print("Model path is not set!!!")        
        
    return tokenizer


def create_model_t5(model_name_or_path):
    
    if model_name_or_path:
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        print("Model path is not set!!!")        
        
    return tokenizer


    
def task_train(args, model, tokenizer, train_data_loader, dev_data_loader, test_data_loader, test_long_loader, optimizer, my_lr_scheduler):
    
    result_name_path = f"../result/{args.model_type}_{args.train_stage}_{args.tuning_mode}_training_samples_{args.training_sample_num}_{args.temperature}.csv"
        
    best_score = 0.0   
    early_stop=0
    coverage = 0.0
    
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
                                    
                    x_token = batch["concept_set_input_ids"].to(args.device).long()
                    input_ids =  batch["cat_text"].to(args.device).long()
                    mask_ids =  batch["mask_ids"].to(args.device).long()                    
                    
                    _,output =  model(x_token, input_ids, mask_ids)
                    loss = output
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

            if epoch+1>=1:
                outpus = run_eval(args, model, dev_data_loader, tokenizer, only_test=True, output_path=args.output_path)
                coverage = outpus["coverage"]
                ppl = outpus["ppl"]
                print("coverage:", coverage)
                print("ppl:", ppl)

            else:
                coverage = 0.0
                continue

            if coverage>best_score:
                early_stop=0
                print("coverage:", coverage)
                best_score = coverage
                outpus_test = run_eval(args, model, test_data_loader, tokenizer, only_test=True, output_path=args.output_path)
                outpus_long = run_eval(args, model, test_long_loader, tokenizer, only_test=False, output_path=args.output_path)
                print(outpus_test)
                print(outpus_long)
            else:
                early_stop+=1
                if early_stop>3:
                    break
                        
    
    in_csv = {"seed":args.seed,"train_stage": args.train_stage,"dev_cov":best_score}
    in_csv.update(outpus_test)
    in_csv.update(outpus_long)
    addCsv(result_name_path, in_csv)
    # save_model(args, model, epoch, round(coverage,2))
    
    
    
    
    
def control_pretrain(args, model, tokenizer, train_data_loader, dev_data_loader, test_data_loader, optimizer, my_lr_scheduler):
    
    result_name_path = f"../result/{args.model_type}_{args.train_stage}_{args.tuning_mode}_training_samples_{args.training_sample_num}_{args.temperature}.csv"
        
    best_score = 0.0   
    early_stop=0
    coverage = 0.0
    
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
                                    
                    x_token = batch["concept_set_input_ids"].to(args.device).long()
                    input_ids =  batch["c_output_ids"].to(args.device).long()
                    
                    _,output =  model(x_token, input_ids)
                    loss = output
                    print("the loss is:", loss)
                    
                    tot_loss += loss.item()

                    loss.backward()
                    torch.cuda.empty_cache()
                    optimizer.step()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
            
                    step += args.batch_size
                    step_count += args.batch_size
                    
                    print(f"epoch is {epoch}, and step size is:{step}")
                    
                    if step_count>args.step_size:
                            outpus = run_eval(args, model, dev_data_loader, tokenizer, only_test=True, output_path=args.output_path)
                            coverage = outpus["coverage"]
                            ppl = outpus["ppl"]
                            print("coverage:", coverage)
                            print("ppl:", ppl)
                            save_model(args, model, step, round(coverage,2))
                            my_lr_scheduler.step()
                            in_csv = {"step":step}
                            in_csv.update(outpus)
                            addCsv(result_name_path,in_csv)
                            step_count=0

    
    
def general_pretrain(args, model, tokenizer, train_data_loader, dev_data_loader, test_data_loader, optimizer, my_lr_scheduler):
    
    
    result_name_path = f"../result/{args.model_type}_{args.train_stage}_{args.tuning_mode}_training_samples_{args.training_sample_num}_{args.temperature}.csv"
        
    best_score = 0.0   
    early_stop=0
    coverage = 0.0
    
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
                                    
                    x_token = batch["concept_set_input_ids"].to(args.device).long()
                    input_ids =  batch["c_output_ids"].to(args.device).long()
                    
                    
                    _,output =  model(x_token, input_ids)
                    loss = output
                    print("the loss is:", loss)
                    
                    tot_loss += loss.item()

                    loss.backward()
                    torch.cuda.empty_cache()
                    optimizer.step()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
            
                    step += args.batch_size
                    step_count += args.batch_size
                    
                    print(f"epoch is {epoch}, and step size is:{step}")
                    
                    if step_count>args.step_size:
                        ppl = run_eval_ppl(args, model, dev_data_loader, tokenizer, only_test=True, output_path=args.output_path)
                        print("ppl:", ppl)
                        save_model(args, model, step, round(ppl,2))
                        my_lr_scheduler.step()

                        in_csv = {"epoch":epoch,"step":step, "ppl":ppl,"loss": round(tot_loss/args.step_size,2)}
                        addCsv(result_name_path,in_csv)

                        tot_loss = 0
                        step_count=0    



if __name__ == "__main__":

    args = construct_generation_args()
    
    
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
    
    
    if args.model_type=="Prompt_Residual_Tuning":
        model = Prompt_Residual_Tuning(args, args.template)
    
    elif args.model_type=="Residual_Tuning":
        model = Residual_Tuning(args, args.template)
        
    elif args.model_type=="Vanilla_Prompt_Tuning":
        model = Vanilla_Prompt_Tuning(args, args.template)
        
    else:
        raise Exception("the task is out of scope!")
    
    if args.check_point_load != None:
        model.prompt_encoder.load_state_dict(load_prompt(args.check_point_load))
        print("load the embedding checkpoint successfully!")
        
    model.to(args.device)
            
    print("args.batch_size:",args.batch_size)
    
    if args.train:
        if args.tuning_mode == "pt":
            params = [{'params': model.prompt_encoder.parameters()}]
            if hasattr(model, 'prompt_encoder_'):
                params.append({'params': model.prompt_encoder_.parameters()})
        else:
            params = [{'params': model.prompt_encoder.parameters()},{'params': model.model.parameters()}]
            if hasattr(model, 'prompt_encoder_'):
                params.append({'params': model.prompt_encoder_.parameters()})
            
            
        optimizer = torch.optim.AdamW(params,  weight_decay= args.weight_decay,lr=args.lr)
        
        
        if args.train_stage == "fine_tuning":
            my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size= int(args.epoch/2), gamma=0.2)
        else:
            my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.5)
            
    
    print("dataset loaded ok!")
    if args.validation or args.test:
        run_eval(args, model, dev_data_loader, tokenizer, only_test=True, output_path=args.output_path)
        
        
    if args.train_stage == "fine_tuning":
        
        train_data = CommonGenDataset(args.train_path, tokenizer, is_training=True, args=args)
        print("train_data:", len(train_data))
        train_data_loader = get_data_loader(train_data, args.batch_size)
        
        dev_data = CommonGenDataset(args.dev_path, tokenizer, is_training=False, args=args)
        dev_data_loader = get_data_loader(dev_data, 100)

        test_data = CommonGenDataset(args.test_path, tokenizer, is_training=False, args=args)
        test_data_loader = get_data_loader(test_data, 100)
        
        long_dis_data = C2Gen(args.long_test_path, tokenizer, args=args)
        test_long_loader = get_data_loader(long_dis_data, 1)
        
        
        task_train(args, model, tokenizer, train_data_loader, dev_data_loader, test_data_loader, test_long_loader, optimizer, my_lr_scheduler)
        
    
    elif  args.train_stage == "general_pretrain":
        
        train_dataset = WikiDataset_General(args.pretrain_path, tokenizer, is_training=True, args=args)
        train_data_loader = get_wiki_data_loader(train_dataset, args.batch_size)

        dev_dataset = WikiDataset_General(args.pretrain_path_val, tokenizer, is_training=True, args=args)
        dev_data_loader = get_wiki_data_loader(dev_dataset, args.batch_size)

        general_pretrain(args, model, tokenizer, train_data_loader, dev_data_loader, None, optimizer, my_lr_scheduler)
        

    elif args.train_stage == "control_pretrain":
        
        train_data = WikiDataset(args.pretrain_path, tokenizer, is_training=True, args=args)
        print("pretrain_train_data:", len(train_data))
        train_data_loader = get_wiki_data_loader(train_data, args.batch_size)
        
        dev_data = CommonGenDataset(args.dev_path, tokenizer, is_training=False, args=args)
        dev_data_loader = get_data_loader(dev_data, 100)
            
        control_pretrain(args, model, tokenizer, train_data_loader, dev_data_loader, None, optimizer, my_lr_scheduler)  
        
    else:
         raise Exception("the task is out of scope!")
        
    

    
    

        
       

