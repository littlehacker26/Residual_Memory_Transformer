from eval_metrics.cider.cider import Cider
# from eval_metrics.spice.spice import Spice
from fast_bleu import BLEU, SelfBLEU
from numpy import *
import numpy as np

import evaluate

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer


import spacy
import sys
import codecs
from  itertools import zip_longest


nlp = spacy.load("en_core_web_sm")

def tokenize(dict):
    for key in dict:
        new_sentence_list = []
        for sentence in dict[key]:
            a = ''
            for token in nlp(sentence):
                a += token.text
                a += ' '
            new_sentence_list.append(a.rstrip())
        dict[key] = new_sentence_list

    return dict


def ppl_from_pretrained_model(logits, labels, attention_mask):
    
    
    bs = logits.shape[0]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_attentions = attention_mask[:, 1:].contiguous()
    
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
    
    loss = loss.mul(shift_attentions.type(torch.uint8))
    
    meanloss = loss.sum(1) / shift_attentions.sum(1)
    ppl = torch.exp(meanloss).cpu().numpy().tolist()
    
    return ppl


def cal_ppl_bygpt2(tokenizer, model, max_length, sentence):
    
    tokenizer.padding_side = "right"
    inputs = tokenizer(sentence, padding='max_length', max_length = max_length, truncation=True, return_tensors="pt").to(model.device)
    bs, sl = inputs['input_ids'].size()
    outputs = model(**inputs, labels=inputs['input_ids'])
    logits = outputs[1]
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs['input_ids'][:, 1:].contiguous()
    shift_attentions = inputs['attention_mask'][:, 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
    
    loss = loss.mul(shift_attentions.type(torch.uint8))
    
    meanloss = loss.sum(1) / shift_attentions.sum(1)
    ppl = torch.exp(meanloss).cpu().numpy().tolist()
    tokenizer.padding_side = "left"

    return ppl


def evaluator_ppl(res, plm_model):
    
    eval_model = AutoModelForCausalLM.from_pretrained(plm_model).cuda()
    eval_tokenizer = AutoTokenizer.from_pretrained(plm_model)
    eval_tokenizer.pad_token = eval_tokenizer.eos_token
    
    ppls = []

    references = [v[0] for k,v in res.items()]
    
    for i in zip_longest(*([iter(references)] * 120), fillvalue= "xxx"):
            i = list(i)
            
            if len(i)<=1:
                continue
            with torch.no_grad():
                ppl = cal_ppl_bygpt2(eval_tokenizer, eval_model, 30, i)
                ppls += ppl
        
    return  np.nanmean(ppls)
        
        

def get_coverage_score(gt_concepts, pred):
    covs = []
    total_cs, match_cs = 0, 0
    for cs, p in zip(gt_concepts, pred):
        p = p.lower()
        if p.endswith('.'):
            p = p[:-1]
            p = p.strip()
        cs = set(cs)
        lemmas = set()
        for token in nlp(p):
            lemmas.add(token.lemma_)
        match_cs += len(lemmas&cs)
        total_cs += len(cs)
        cov = len(lemmas&cs)/len(cs)
        covs.append(cov)
    return 100 * sum(covs) / len(covs), 100 * match_cs / total_cs



def evaluator_coverage(res):
    
    pred = tokenize(res)
    gt_concepts = [r.split('#') for r in list(pred.keys())]
    
    predictions = [v[0] for k,v in pred.items()]

    score = get_coverage_score(gt_concepts, predictions)
    
    return mean(score)
    
    
    
    
    
    
    
    
    

def evaluator_cider(gts, res):
    eval = {}
    # =================================================
    # Set up scorers
    # =================================================
    # Todo: use Spacy for tokenization
    gts = tokenize(gts)
    res = tokenize(res)
       
    score, scores = Cider().compute_score(gts, res)
    
    
    return score


def evaluator_meteor(gts, res):
    
    predictions = [v[0] for k,v in res.items()]
    references = [v for k,v in gts.items()]
    
    meteor = evaluate.load('meteor')
   
    results = meteor.compute(predictions=predictions, references=references)
    
    return round(results['meteor'], 2)


    



def evaluator_bleu(gts, res):
    
    bigram = []
    trigram = []
    quagram = []
    
    predictions = [v[0] for k,v in res.items()]
    references = [v for k,v in gts.items()]
    
    weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.), 'quagram': (1/4., 1/4., 1/4.,1/4.)}

    for p, ref in zip(predictions,references):
        list_of_references = [r.split() for r in ref]
        hypotheses = [p.split()]
        bleu = BLEU(list_of_references, weights)
        score = bleu.get_score(hypotheses)
        bigram.append(score["bigram"][0])
        trigram.append(score["trigram"][0])
        quagram.append(score["quagram"][0])
        
    return {'bigram':mean(bigram), 'trigram':mean(trigram), 'quagram':mean(quagram)}



def evaluator_selfbleu(res):
    
    bigram = []
    trigram = []
    quagram = []
    
    predictions = [v[0] for k,v in res.items()]
    hypotheses = [i.split() for i in predictions]

    weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.), 'quagram': (1/4., 1/4., 1/4.,1/4.)}

    self_bleu = SelfBLEU(hypotheses, weights)
    score = self_bleu.get_score()
    return {'bigram':mean(score['bigram']), 'trigram':mean(score['trigram']), 'quagram':mean(score['quagram'])}



if __name__=='__main__':


    gts = {"cat#dog#boy": ["The dog is the boy's cat.", "The dog eats the cat of the boy."],
           "apple#tree#boy": ["A boy is picking apples from trees."]}
    
    res = {"cat#dog#boy": ["The dog is the boy's cat."],
           "apple#tree#boy": ["A boy is picking apples from trees and put them into bags."]}
    
    a = evaluator_coverage(res)
    print("Coverage score is:", a)
    
    score = evaluator_cider(gts, res)
    print("Cider: %0.3f" %score)
    
    score = evaluator_meteor(gts, res)
    print("Meteor: %0.3f" %score)
    
    score = evaluator_bleu(gts, res)
    print("Bleu:", score)
    
    score = evaluator_selfbleu(res)
    print("self bleu:", score)
    
    score = evaluator_ppl(res, "/home2/zhanghanqing/pretrained_model/gpt2/large")
    print("PPL: %0.3f" %score)
    
    
   
    
    
