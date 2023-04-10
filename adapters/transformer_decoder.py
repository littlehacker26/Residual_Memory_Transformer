
import torch
import torch.nn as nn
import torch.nn.functional as F


import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
from torch.nn import MultiheadAttention
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm




class Transformer_Decoder(nn.TransformerDecoderLayer):
    
        
        
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        
        
        super().__init__(d_model, nhead, dim_feedforward, dropout,
                 activation, layer_norm_eps, batch_first, norm_first,
                 device, dtype)
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        
        self.self_attn_casual = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        

        
        self.norm4 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout4 = Dropout(dropout)

        
        
    def _sa_block_casual(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        
        
        x = self.self_attn_casual(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout4(x)
    
    

    
    def forward(self, tgt: Tensor, tgt_: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
            r"""Pass the inputs (and mask) through the decoder layer.

            Args:
                tgt: the sequence to the decoder layer (required).
                memory: the sequence from the last layer of the encoder (required).
                tgt_mask: the mask for the tgt sequence (optional).
                memory_mask: the mask for the memory sequence (optional).
                tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
                memory_key_padding_mask: the mask for the memory keys per batch (optional).

            Shape:
                see the docs in Transformer class.
            """
            # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

            x = tgt
            x1 = tgt_
            
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                x = x +  self._sa_block_casual(self.norm4(x1), tgt_mask, tgt_key_padding_mask)
                x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                x = x + self._ff_block(self.norm3(x))
            else:
                x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
                x = self.norm4(x+ self._sa_block_casual(x1, tgt_mask, tgt_key_padding_mask))
                x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
                x = self.norm3(x + self._ff_block(x))

            return x



            
        