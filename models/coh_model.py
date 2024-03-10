from transformers import  BertModel, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.discriminator import MLP_Discriminator


class Coh_base(torch.nn.Module):
    def __init__(self):
        super(Coh_base, self).__init__()
        self.embed_dim = 768
      
        self.discriminator = MLP_Discriminator(self.embed_dim)
      
        
        
    def forward(self, anchor_ids,sent_ids):
        #Encode tokens
        scores = self.discriminator(anchor_ids,sent_ids)
        
        return scores


