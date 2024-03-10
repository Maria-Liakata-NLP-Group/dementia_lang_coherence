import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration


class T5(torch.nn.Module):
    def __init__(self):
        super(T5, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base",bos_token='<startoftext>', eos_token='<endoftext>')
        
        self.T5 = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.T5.resize_token_embeddings(len(self.tokenizer))
        
    def forward(self, input_ids, attention_masks, decoder_input_ids,decoder_attention_mask):
        t5 = self.T5(input_ids = input_ids, attention_mask=attention_masks, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
        output = t5[0]
    
        return output
            