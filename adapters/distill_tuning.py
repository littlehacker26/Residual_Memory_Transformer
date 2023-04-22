import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering

import torch
from torch.nn import GRU


import re
import datetime

from transformers import AutoTokenizer

from .models import get_embedding_layer, create_model, _create_model
from .prompt_encoder import PromptEncoder
from .transformer_decoder import Transformer_Decoder

import torch
import torch.nn as nn
import torch.nn.functional as F

SMALL_CONST = 1e-10
BIG_CONST = -1e15




class Residual_Model(nn.Module):
    def __init__(self, input_size, n_head, n_layer_encoder, n_layer_decoder):
        super(Residual_Model, self).__init__()
        
        self.encoder_block = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=input_size, nhead=n_head, batch_first=True) for i in range(n_layer_encoder)]
        )
            
        self.decoder_block = nn.ModuleList(
            [Transformer_Decoder(d_model=input_size, nhead=n_head, batch_first=True) for i in range(n_layer_decoder)]
        )
        
        self.lm_head = nn.Linear(input_size, 50257,bias=False)
                
        
    def forward(self, tgt, memory, tgt_mask, memory_mask, att_mask):
        
        for layer_module in  self.encoder_block:
            memory = layer_module(src = memory, src_key_padding_mask = memory_mask)
            
        inp = tgt[0]
        
        for layer_module in  self.decoder_block:
            inp = layer_module(tgt =inp, tgt_ =tgt[-1], memory=memory, tgt_key_padding_mask = tgt_mask, memory_key_padding_mask=memory_mask, tgt_mask=att_mask)
 
        return self.lm_head(inp)

    
    
class Distill_Tuning(torch.nn.Module):

    def __init__(self, args, template):
        super().__init__()
        self.args = args

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # model setting
        self.model = create_model(self.args)
        
        
        if self.args.tuning_mode == "pt":
            for param in self.model.parameters():
                param.requires_grad = False
            
            
        self.position_embedings = self.model.transformer.wpe
        self.position_embedings.requires_grad = False
        
        self.embeddings = self.model.get_input_embeddings()
        self.embeddings.requires_grad = False
        
        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        print("layer of resiudal:", self.args.residual_layer)
        
        self.prompt_encoder = Residual_Model(input_size = self.hidden_size, n_head = 8, n_layer_encoder= self.args.residual_layer, n_layer_decoder= self.args.residual_layer)
        

#     @torch.no_grad()
#     def get_gpt_embeddings(self, input_ids, position_ids):
        
#         output_decoder = self.model(input_ids=input_ids,
#                                     position_ids=position_ids,
#                                     output_hidden_states = True,
#                                     return_dict= True)
#         decoder_hidden = output_decoder.hidden_states[-1] #batch*seq*hidden
        
#         return decoder_hidden.detach()
    
    
    @torch.no_grad()
    def get_gpt_embeddings(self, input_ids, position_ids):
        
        position_embeds = self.position_embedings(position_ids)
        inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds + position_embeds 
        return hidden_states.detach()
                
    
    def load_prompt(self, embedding_checkpoint):
        checkpoint = torch.load(embedding_checkpoint)
        prompt_embedding = checkpoint['embedding']
        return prompt_embedding    
    
    
    def prepare_decoder_input_ids(self, control_ids= None, target_ids = None):
        
        start_of_token =  torch.zeros(control_ids.shape[0], 1).to(self.args.device).fill_(self.tokenizer.pad_token).long()
        
        if target_ids == None:
            return torch.cat([control_ids, start_of_token], dim=1)
        else:
            return torch.cat([control_ids,start_of_token,target_ids], dim=1)
        
                    
    def _generate_square_subsequent_mask(self, sz: int, device='cpu'):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        # return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return torch.triu(torch.full((sz, sz), True, device=device), diagonal=1)

    
    def KL_loss(self, input_x, input_y, attention_mask):
        """
        compute the KL loss
        """
        m = torch.flatten(attention_mask)
        indices = torch.nonzero(m).squeeze(-1)
        
        x = input_x.reshape(-1,input_x.shape[-1])
        x = torch.index_select(x, 0, indices)
            
        y = input_y.reshape(-1,input_y.shape[-1])
        y = torch.index_select(y, 0, indices)
        
        input_data = F.log_softmax(x, dim=1)
        target = F.softmax(y, dim=1)
        loss = self.kl_loss(input_data, target)
        
        return  loss
    
    
    def forward(self, x_hs, x_ts):

        control_input_ids = x_hs
        attention_mask_control = (control_input_ids!= self.tokenizer.pad_token_id).to(x_hs.device).bool()
        position_ids_control = attention_mask_control.long().cumsum(-1)-1

        control_hidden = self.get_gpt_embeddings(control_input_ids, position_ids_control)
        
        context = x_ts
        attention_mask = (x_ts!= self.tokenizer.pad_token_id).bool()
        
        labels = torch.clone(context)
        labels.masked_fill_(attention_mask==0, -100)
        
        with torch.no_grad():
        
            output_decoder = self.model(input_ids=context,
                                        attention_mask=attention_mask,
                                        output_hidden_states = True,
                                        return_dict= True)
            decoder_hidden = output_decoder.hidden_states #batch*seq*hidden
            
        
        att_mask = self._generate_square_subsequent_mask(decoder_hidden[0].shape[1],self.args.device).bool()
      
        logits = self.prompt_encoder(tgt=decoder_hidden, memory=control_hidden, tgt_mask=~attention_mask, memory_mask=~attention_mask_control, att_mask=att_mask)

        shift_loigt = logits[:,:-1,:].reshape(-1, logits.shape[-1])
        shift_label = labels[:,1:].reshape(-1)
        
        loss = self.loss_fct(shift_loigt, shift_label)
        
        # kl_loss = self.KL_loss(logits, output_decoder.logits, attention_mask)
                    
        return logits, loss #+kl_loss