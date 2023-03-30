import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering

import torch
from torch.nn import GRU
from transformers import GPT2LMHeadModel


import re
import datetime

from transformers import AutoTokenizer

from .models import get_embedding_layer, create_model, _create_model
from .prompt_encoder import PromptEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


SMALL_CONST = 1e-10
BIG_CONST = -1e15

    

class GPT2_Tuning(GPT2LMHeadModel):
    
    def __init__(self, config):
        super().__init__(config)
        

    def init_post(self, args):
        
        self.args = args
        # load tokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
        decoder_input_ids = input_ids
        attention_mask = (decoder_input_ids!= self.tokenizer.pad_token_id).to(decoder_input_ids.device).bool()
        
        output_decoder = super().forward(input_ids = input_ids,
                                         past_key_values= past_key_values,
                                         attention_mask = attention_mask,
                                         token_type_ids = None,
                                         position_ids = position_ids,
                                         head_mask = head_mask,
                                         inputs_embeds = inputs_embeds,
                                         encoder_hidden_states = None,
                                         encoder_attention_mask=encoder_attention_mask,
                                         labels = None,
                                         use_cache = use_cache,
                                         output_attentions=use_cache,
                                         output_hidden_states=True,
                                         return_dict=True)
        
        logits = output_decoder.logits
        
        if self.training:
            labels = torch.clone(decoder_input_ids)
            labels.masked_fill_(token_type_ids==0, -100)
            labels.masked_fill_(attention_mask==False, -100)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_loigt = logits[:,:-1,:].reshape(-1, logits.shape[-1])
            shift_label = labels[:,1:].reshape(-1)
            loss = loss_fct(shift_loigt, shift_label)
        else:
            loss = None
        
        output_decoder.loss = loss
        
        return output_decoder
            
    
    
    
