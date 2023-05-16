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


from .disc import DISC


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


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
        
        self.prompt_encoder = Residual_Model(input_size = self.hidden_size, n_head = 8, n_layer_encoder= self.args.residual_layer, n_layer_decoder= self.args.residual_layer)
        
        for param in self.transformer.parameters():
                param.requires_grad = False
            
        self.disc = DISC(args)
        
    
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
    
    
    
    
    def top_k_top_p_filtering(self,
        logits,
        top_k = 0,
        top_p = 1.0,
        filter_value = -1e15 ,
        min_tokens_to_keep = 1,
    ):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits
        
    
    
    
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

                beam_size = int(input_ids.shape[0]/encoder_data.shape[0])
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
            
        y_ = torch.softmax(y/self.args.temperature, dim = -1)
        loss_ = -(y_ * (x+1e-20).log()).sum() / x.size(0)
        
        _y = torch.softmax(((1-y).mul(y>0.0))/self.args.temperature, dim = -1)
        _loss = -(_y * (1-x+1e-20).log()).sum() / x.size(0)

        return  loss_ + _loss

    
    
    
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
        
        
        if self.training:
            
            control_input_ids = labels
            _attention_mask_control = (control_input_ids!= self.tokenizer.pad_token_id).to(labels.device).bool()
            position_ids_control = attention_mask_control.long().cumsum(-1) - 1
            position_ids_control = torch.clamp(position_ids_control, min=0)
            _control_hidden = self.get_gpt_embeddings(control_input_ids, position_ids_control)
            
        
                
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
            _logits = self.args.memory_p*self.prompt_encoder(tgt=decoder_hidden, memory=_control_hidden, tgt_mask=~attention_mask, memory_mask=~_attention_mask_control, att_mask=att_mask) + (1-self.args.memory_p)*output_decoder.logits
            
                
        if self.training:
            
            logits_candidate = output_decoder.logits
            logits_candidate = self.top_k_top_p_filtering(logits_candidate.view(logits_candidate.shape[0]*logits_candidate.shape[1], -1), top_k= self.args.ranking_scope , top_p=self.args.top_p, filter_value=BIG_CONST).view(output_decoder.logits.shape[0], output_decoder.logits.shape[1], -1)

            reank_output = self.disc.get_ranked_logtis(input_ids, logits_candidate.detach().clone(), desired_att=None)
        
            reank_output_pos = (logits_candidate>BIG_CONST+10).mul(reank_output)
            pos_loss = self.KL_loss(torch.softmax(logits, dim=-1), reank_output_pos, attention_mask)
            
            reank_output_neg = (logits_candidate>BIG_CONST+10).mul(1-reank_output)
            neg_loss = self.KL_loss(torch.softmax(_logits, dim=-1), reank_output_neg, attention_mask)

            loss= pos_loss+neg_loss
        else:
            loss = None
        
        output_decoder.loss = loss
        output_decoder.logits = logits
        
        return output_decoder