from tqdm import *
from transformers import  BertTokenizer,RobertaTokenizer
import torch
import pandas as pd
import pickle
import os
from torch.utils.data import TensorDataset
from random import shuffle, random, randrange
from transformers import T5Tokenizer

class Extract_features():
     def __init__(self, data_path: str, model_name = str, task = str, negative_samples = str,  max_seq_length=512):
         self.data_path = data_path
         self.max_seq_length = max_seq_length
         self.model_name  = model_name
         self.task = task
         self.negative_samples = negative_samples
         
         if self.model_name=='bert':
             self.tokenizer =BertTokenizer.from_pretrained('bert-large-uncased')
         elif self.model_name=='roberta':
             self.tokenizer =RobertaTokenizer.from_pretrained('roberta-base', max_length = 512)   
         elif self.model_name=='t5-base':    
             self.tokenizer = T5Tokenizer.from_pretrained(self.model_name,sep_token='<sep>')
             
         
     def tokenize(self):
         df = pd.read_csv(self.data_path, sep=',')
         
         map_label = {0:'negative', 1:'positive'}
         
         if self.model_name=='t5-base':    
             df['Label'] = df['Label'].apply(lambda x:map_label[x])
         
         input_ids = []
         attention_masks = []
         type_ids = []
         labels = []
         labels_mask = []
         
         
         for index, row in tqdm(df.iterrows(), total=df.shape[0]):
             sen1 = df.iloc[index]['Anchor'].lower()
             sen2 = df.iloc[index]['Sentence'].lower()
             
             tokens1 = self.tokenizer.tokenize(sen1)
             tokens2 = self.tokenizer.tokenize(sen2)
             
             self._truncate_seq_pair(tokens1, tokens2, self.max_seq_length - 3)
             
             if self.model_name=='bert':
                 tokens = [self.tokenizer.cls_token] + tokens1 + [self.tokenizer.sep_token]
             elif self.model_name=='roberta':
                 tokens = ["</s>"] + tokens1 + ["</s>"]
             elif self.model_name=='t5-base':      
                 tokens = tokens1 + [self.tokenizer.sep_token]
             segment_ids = [0] * len(tokens)
             
             if tokens2:
                 if self.model_name=='bert':
                     tokens += tokens2 + + [self.tokenizer.sep_token]
                     segment_ids += [1] * (len(tokens2)+1)
                 elif self.model_name=='roberta': 
                     tokens += tokens2 + ["</s>"]
                     segment_ids += [1] * (len(tokens2)+1)
                 elif self.model_name=='t5-base':    
                     tokens += tokens2
                     segment_ids += [1] * (len(tokens2))
                 
                 
# =============================================================================
#                  
#              encoded_dict = self.tokenizer.encode_plus(
#                             tokens,  # document to encode.
#                             add_special_tokens=False,  # add '[CLS]' and '[SEP]'
#                             padding='max_length',  # set max length
#                             truncation=True,  # truncate longer messages
#                             pad_to_max_length=True,  # add padding
#                             return_attention_mask=True,  # create attn. masks
#                             return_tensors='pt'  # return pytorch tensors
#                             )
# =============================================================================
             

            

             
             encoded_ids = self.tokenizer.convert_tokens_to_ids(tokens)
             encoded_mask = len(encoded_ids)*[1]
             
             
             # Zero-pad up to the sequence length.
             padding = [0] * (self.max_seq_length - len(encoded_ids))
             encoded_ids += padding
             encoded_mask += padding
             segment_ids += padding
             assert len(encoded_ids) == self.max_seq_length
             assert len(encoded_mask) == self.max_seq_length
             assert len(segment_ids) == self.max_seq_length
             
             input_ids.append(torch.tensor(encoded_ids))
             attention_masks.append(torch.tensor(encoded_mask))
             type_ids.append(torch.tensor(segment_ids))
             
             #add labels
             if self.model_name=='bert' or self.model_name=='roberta':
                 labels.append(df.iloc[index]['Label'])
             elif self.model_name=='t5-base':
                 decoded_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(df.iloc[index]['Label']))
                 decoded_mask = len(decoded_ids)*[1]
                 padding = [0] * (self.max_seq_length - len(decoded_ids))
                 decoded_ids += padding
                 decoded_mask+= padding
                 labels.append(torch.tensor(decoded_ids))
                 labels_mask.append(torch.tensor(decoded_mask))
             
         # Convert the lists into tensors.
         if self.model_name=='bert' or self.model_name=='roberta':
             input_ids = torch.stack(input_ids, dim=0)
             attention_masks = torch.stack(attention_masks, dim=0)
             type_ids = torch.stack(type_ids, dim=0)
             labels = torch.tensor(labels)    
         elif self.model_name=='t5-base':  
             input_ids = torch.stack(input_ids, dim=0)
             attention_masks = torch.stack(attention_masks, dim=0)
             labels = torch.stack(labels, dim=0)
             labels_mask = torch.stack(labels_mask, dim=0)
         
         if self.model_name=='bert' or self.model_name=='roberta':   
             dataset = TensorDataset(input_ids, attention_masks, type_ids, labels)
         elif self.model_name=='t5-base':      
             dataset = TensorDataset(input_ids, attention_masks, labels, labels_mask)
             
         if 'train' in self.data_path:
             subset = 'train'
         elif 'test'in self.data_path:
             subset = 'test'
         elif 'val' in self.data_path:
             subset = 'val'
            
         with open(os.path.join('../data', self.task + '_' + subset +  '_' + negative_samples + '.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
            
     def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
           """Truncates a sequence pair in place to the maximum length."""
           # This is a simple heuristic which will always truncate the longer sequence
           # one token at a time. This makes more sense than truncating an equal percent
           # of tokens from each, since if one sequence is very short then each token
           # that's truncated likely contains more information than a longer sequence.
           while True:
               total_length = len(tokens_a) + len(tokens_b)
               if total_length <= max_length:
                   break
               if len(tokens_a) > len(tokens_b):
                   tokens_a.pop()
               else:
                  tokens_b.pop()       
         
         
         
if __name__ == "__main__":   
    data_path = 'data/nsp_train_same_.csv'
    negative_samples = 'same' #Could be either same or across 
    model_name = 'roberta' #can be either roberta, bert, t5-base, or t5-large
    task = 'nsp'    #can be either nsp or 
    max_seq_length = 512
   
    
    self = Extract_features(data_path,model_name, task, negative_samples, max_seq_length)
    self.tokenize()                 
                         
    


