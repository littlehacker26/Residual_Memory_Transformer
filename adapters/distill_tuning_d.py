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
from .transformer_decoder import Transformer_Decoder


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


SMALL_CONST = 1e-10
BIG_CONST = -1e15



class Residual_Model(nn.Module):
    def __init__(self, input_size, n_head, n_layer):
        super(Residual_Model, self).__init__()
        
        self.encoder_block = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=input_size, nhead=n_head, batch_first=True) for i in range(n_layer)]
        )
            
        self.decoder_block = nn.ModuleList(
            [Transformer_Decoder(d_model=input_size, nhead=n_head, batch_first=True) for i in range(n_layer)]
        )
        
        self.lm_head = nn.Linear(input_size, 50257,bias=False)
                
        
    def forward(self, tgt, memory, tgt_mask, memory_mask, att_mask):
        
        for layer_module in  self.encoder_block:
            memory = layer_module(src = memory, src_key_padding_mask = memory_mask)
            
        inp = tgt[0]
        
        for layer_module in  self.decoder_block:
            inp = layer_module(tgt =inp, tgt_ =tgt[-1], memory=memory, tgt_key_padding_mask = tgt_mask, memory_key_padding_mask=memory_mask, tgt_mask=att_mask)
 
        return self.lm_head(inp)

    

class Distill_Tuning(GPT2LMHeadModel):
    
    def __init__(self, config):
        super().__init__(config)
        

    def init_post(self, args):
        
        self.args = args
        # load tokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
                
        self.embeddings = self.get_input_embeddings()
        
        self.position_embedings = self.transformer.wpe
        
        self.hidden_size = self.embeddings.embedding_dim      
        self.prompt_encoder = Residual_Model(input_size = self.hidden_size, n_head = 8, n_layer=self.args.residual_layer)
        self.transformer.requires_grad = False
        
        
    
    def load_prompt(self, embedding_checkpoint):
        checkpoint = torch.load(embedding_checkpoint)
        prompt_embedding = checkpoint['embedding']
        return prompt_embedding    
    
                    
    def _generate_square_subsequent_mask(self, sz: int, device='cpu'):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        # return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return torch.triu(torch.full((sz, sz), True, device=device), diagonal=1)
    
    
    @torch.no_grad()
    def get_gpt_embeddings(self, input_ids, position_ids):
                        
        position_embeds = self.position_embedings(position_ids)
        inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds + position_embeds 
        
        return hidden_states.detach()
    
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past_key_values is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        
                    
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None


        if "encoder_hidden_states" in kwargs:  # we only want to use them in the 1st generation step

            encoder_data = kwargs.get("encoder_hidden_states", None)

            if encoder_data.shape[0] != input_ids.shape[0]:

                # print("input_ids:",input_ids.shape)
                beam_size = int(input_ids.shape[0]/encoder_data.shape[0])
                # print("beam_size:", beam_size)
                encoder_data = encoder_data.repeat_interleave(beam_size, dim=0)

            model_inputs = {"encoder_hidden_states": encoder_data}
            


        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "input_ids":input_ids
        })
        # print(model_inputs["encoder_hidden_states"])
        return model_inputs
    
    
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
        
        control_input_ids = encoder_hidden_states
        attention_mask_control = (control_input_ids!= self.tokenizer.pad_token_id).to(encoder_hidden_states.device).bool()
        position_ids_control = attention_mask_control.long().cumsum(-1) - 1
        position_ids_control = torch.clamp(position_ids_control, min=0)
        control_hidden = self.get_gpt_embeddings(control_input_ids, position_ids_control)
        
        
        
        if input_ids!= None and self.training:
            eos_musk = torch.ones(control_input_ids.shape[0], 1).to(self.args.device).bool()
            attention_mask = (input_ids!= self.tokenizer.pad_token_id).bool()
            attention_mask = torch.cat([eos_musk,attention_mask], dim=1).bool()
            
            eos = torch.zeros(control_input_ids.shape[0], 1).to(self.args.device).fill_(self.tokenizer.pad_token_id).long()
            input_ids = torch.cat([eos,input_ids], dim=1)
                        
        
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
        
        decoder_hidden = output_decoder.hidden_states #batch*seq*hidden
        
        if self.training:
            att_mask = self._generate_square_subsequent_mask(decoder_hidden[0].shape[1],self.args.device).bool()
        else:
            att_mask = None
        
        logits = self.args.memory_p*self.prompt_encoder(tgt=decoder_hidden, memory=control_hidden, tgt_mask=~attention_mask, memory_mask=~attention_mask_control, att_mask=att_mask) + (1-self.args.memory_p)*output_decoder.logits
        
        if self.training:
            labels = torch.clone(input_ids)
            labels.masked_fill_(attention_mask==False, -100)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_loigt = logits[:,:-1,:].reshape(-1, logits.shape[-1])
            shift_label = labels[:,1:].reshape(-1)
            loss = loss_fct(shift_loigt, shift_label)
        else:
            loss = None
        
        output_decoder.loss = loss
        output_decoder.logits = logits
        
        return output_decoder
            
    
    
    
