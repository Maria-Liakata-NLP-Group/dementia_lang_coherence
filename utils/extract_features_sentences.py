from tqdm import *
from transformers import  BertTokenizer,RobertaTokenizerFast
import torch
import pandas as pd
import pickle
import os
from torch.utils.data import TensorDataset
from random import shuffle, random, randrange


class Extract_features():
     def __init__(self, data_path: str, model_name = str, task = str, negative_samples = str,  max_seq_length=512):
         self.data_path = data_path
         self.max_seq_length = max_seq_length
         self.model_name  = model_name
         self.task = task
         self.negative_samples = negative_samples
         
         self.tokenizer =BertTokenizer.from_pretrained('bert-large-uncased') 
         
     def tokenize(self):
         df = pd.read_csv(self.data_path, sep=',')
         
         input_ids_1 = []
         attention_masks_1 = []
         input_ids_2 = []
         attention_masks_2 = []
        
         labels = []
         
         
         for index, row in tqdm(df.iterrows(), total=df.shape[0]):
             sen1 = df.iloc[index]['Utter1'].lower()
             sen2 = df.iloc[index]['Utter2'].lower()
             
             tokens1 = self.tokenizer.tokenize(sen1)
             tokens2 = self.tokenizer.tokenize(sen2)
             
            
                 
             encoded1_ids = self.tokenizer.convert_tokens_to_ids(tokens1)
             encoded1_mask = len(encoded1_ids)*[1]
             # Zero-pad up to the sequence length.
             padding = [self.tokenizer.pad_token_id] * (self.max_seq_length - len(encoded1_ids))
             encoded1_ids += padding
             encoded1_mask += padding
             assert len(encoded1_ids) == self.max_seq_length
             assert len(encoded1_mask) == self.max_seq_length
             
             encoded2_ids = self.tokenizer.convert_tokens_to_ids(tokens2)
             encoded2_mask = len(encoded2_ids)*[1]
             # Zero-pad up to the sequence length.
             padding = [self.tokenizer.pad_token_id ] * (self.max_seq_length - len(encoded2_ids))
             encoded2_ids += padding
             encoded2_mask += padding
             assert len(encoded2_ids) == self.max_seq_length
             assert len(encoded2_mask) == self.max_seq_length


             
             input_ids_1.append(torch.tensor(encoded1_ids))
             attention_masks_1.append(torch.tensor(encoded1_mask))
             input_ids_2.append(torch.tensor(encoded2_ids))
             attention_masks_2.append(torch.tensor(encoded2_mask))
             
             
             #add labels
             labels.append(df.iloc[index]['Label'])
             
         # Convert the lists into tensors.
         input_ids_1 = torch.stack(input_ids_1, dim=0)
         attention_masks_1 = torch.stack(attention_masks_1, dim=0)
         input_ids_2 = torch.stack(input_ids_2, dim=0)
         attention_masks_2 = torch.stack(attention_masks_2, dim=0)
         labels = torch.tensor(labels)    
         
         dataset = TensorDataset(input_ids_1, attention_masks_1, input_ids_2, attention_masks_2,labels)
             
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
    data_path = '../data/nsp_test_same_.csv'
    negative_samples = 'same' #Could be either same or across 
    model_name = 'bert' #can be either roberta or bert
    task = 'nsp'    
    max_seq_length = 512
   
    
    self = Extract_features(data_path,model_name, task, negative_samples, max_seq_length)
    self.tokenize()                 
                         
    


