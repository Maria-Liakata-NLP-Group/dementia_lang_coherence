import os
import random
import numpy as np
import torch
import torch.nn as nn
from argument_parser import parse_arguments
from models.bert import Bert
from models.roberta import Roberta
from models.bert_cnn import BertCNN
from models.cnn import CNN
from models.bert_small import Bert_base
from load_data import Load_Data
from torch.optim import AdamW,Adam
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup
from tqdm import tnrange, tqdm
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_loader import Dataload
from torch.utils.data import DataLoader
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


class Extract_feautres(nn.Module):
    def __init__(self,  model):
        super(Extract_feautres, self).__init__()
        self.model = model
        
        if 'gpt' in self.model:
            self.tokenizer  =GPT2Tokenizer.from_pretrained(self.model)
        elif 't5' in self.model:    
            self.tokenizer  =T5Tokenizer.from_pretrained(self.model)
        else:
            print('Undefined model')
    def forward(self, anchors, sentences):
        
        ids = []
        masks = []
        for i,anchor in enumerate(anchors):
            
            sen1 = anchor
            sen2 = sentences[i]
            
            pair = sen1  + sen2
            
            _input = self.tokenizer(pair, return_tensors="pt")
       
        return _input



def test(args,source):
    
    model_bin = [args.task, args.model_name, args.train_task, args.negative_samples, args.batch_size, args.gradient_accumulation_steps, args.lr, 'train' ]
    
    
    if 'gpt2' in args.model_name:
        model =  GPT2LMHeadModel.from_pretrained(args.model_name, return_dict=True)
    elif 't5' in args.model_name:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        print('Undefined model')
    print(model)        
       
       
        
    print('You use only one device')  
    device = torch.device("cuda" if USE_CUDA else "cpu")
    model.to(device)
       
        
     
        
    print('The model has {:} different named parameters.\n'.format(len(list(model.named_parameters()))))
    
    features = Extract_feautres(args.model_name)
    
    df_session = pd.DataFrame(columns = ['client','session', 'f_pos','f_neg', 'acc'])
    index_session = 0
   
    model.eval()
    for subdir , dirs, files in os.walk(source): 
        for file in tqdm(files):
            file_path = os.path.join(subdir, file)
            if file.endswith(".txt"):
                
                
                client = file.split('.')[0].split('-')[0].split('_')[1]
                session = file.split('.')[0].split('-')[1]
                f = open(os.path.join(source, file), 'r')
                seq = f.readlines()
                
                df_data = pd.DataFrame(columns = ['Client','Session', 'Anchor','Sentence', 'Label'])
                index = 0
                
                #Create possitive samples
                for i in range(len(seq) -2):
                    anchor = seq[i].replace('\n', '')
                    #print(anchor)
                    Sentence =seq[i+1].replace('\n', '')
                    df_data.loc[index] = pd.Series({'Client':int(client), 'Session':int(session), 'Anchor':anchor, 'Sentence':Sentence, 'Label':1})
                    index+=1
                    
                #Create negative samples 
                for i in range(len(seq)):
                    acnhor = seq[i].replace('\n', '')
                    for j in range(len(seq)):
                        if  j>i+1:
                            Sentence = seq[j].replace('\n', '')
                            df_data.loc[index] = pd.Series({'Client':int(client), 'Session':int(session), 'Anchor':acnhor, 'Sentence':Sentence, 'Label':0})
                            index+=1   
                            
                test_data = Dataload(df_data)
                test_loader = DataLoader(dataset=test_data, batch_size=1,shuffle=False,drop_last=False)            
                
                f_pos = 0
                f_neg = 0
                pos_case = []
                neg_case = []
                acc = 0
                inferences = []
                test_loss = 0.0
                
                
                for step, (anchor, pos, neg) in tqdm(enumerate(test_loader)):
                    
                    pos_ids = features(anchor,pos)
                    target_pos = features(anchor,pos)
                    neg_ids = features(anchor,neg)
                    target_neg = features(anchor,neg)
                    #Load data
                    pos_ids = pos_ids.to(device)
                    target_pos = target_pos.to(device)
                    neg_ids  = neg_ids.to(device)
                    target_neg = target_neg.to(device)
                        
                        
                    
                    with torch.no_grad():  
                        outputs = model(pos_ids['input_ids'], labels=target_pos['input_ids'])
                        log_likelihood = outputs.loss
                        pos_scores = [torch.exp(log_likelihood)]
                        pos_scores = pos_scores[0].cpu().detach().numpy().flatten().tolist()
                        pos_scores = [1 - score for score in pos_scores]
                        f_pos += sum(pos_scores)
                        pos_case.append(pos_scores[0])
                        
                        outputs = model(neg_ids['input_ids'], labels=target_neg['input_ids'])
                        log_likelihood = outputs.loss
                        neg_scores = [torch.exp(log_likelihood)]
                        neg_scores = neg_scores[0].cpu().detach().numpy().flatten().tolist()
                        neg_scores = [1 - score for score in neg_scores]
                        f_neg += sum(neg_scores)
                        neg_case.append(neg_scores[0])
                        
        
                         
                    for i in range(len(pos_scores)):
                        if pos_scores[i]>neg_scores[i]:
                            inferences.extend([1])
                        else:
                            inferences.extend([0])
                
                f_pos = f_pos/(len(seq) -2)     
                f_neg = f_neg/(len(seq) -2)    
                acc = inferences.count(1)/len(inferences)                     
                df_session.loc[index_session] = pd.Series({'client':int(client), 'session':int(session), 'f_pos':f_pos, 'f_neg':f_neg, 'acc':acc})
                index_session +=1
                
                #print('\n')
                #print(pos_case)
                #print('\n')
                #print(neg_case)
    
    df_session.to_csv('best/'+'_'.join([str(i) for i in model_bin]) + '.csv')
    
if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    source = 'data/test'
  
    USE_CUDA = torch.cuda.is_available()
    test(args,source)
