import os
import random
import numpy as np
import torch
import torch.nn as nn
from argument_parser import parse_arguments
from load_data import Load_Data
from models.coh_model import Coh_base
from load_data import Load_Data
from torch.optim import AdamW,Adam
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup
from tqdm import tnrange, tqdm
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
from data_loader import Dataload
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import pickle
from scipy.special import expit
from transformers import BertModel,BertTokenizer
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

class MarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, p_scores, n_scores, weights=None):
        scores = self.margin - p_scores + n_scores
        scores = scores.clamp(min=0)
        if weights is not None:
            scores = weights * scores
        return scores.mean()
    
    
class Sentence_features(nn.Module):
    def __init__(self):
        super(Sentence_features, self).__init__()
        self.model = SentenceTransformer('sentence-transformers/distilroberta-base-msmarco-v1')     
    
        
    def forward(self, sentences):
        
        ids = []
        for sentence in sentences:
            ids.append(torch.tensor(self.model.encode(sentence)))
        ids = torch.stack(ids, dim=0)
        return ids
    
class Word_features(nn.Module):
    def __init__(self):
        super(Word_features, self).__init__()
        self.tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
    def forward(self, sentences):
        
        ids = []
        for sentence in sentences:
            tokenized_text = self.tokenizer.tokenize(sentence)
            sent_len = len(tokenized_text)
            tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens = tokens + [0]*(512-len(tokens))
            tokens_tensor = torch.tensor(tokens)
            attention_mask = torch.tensor(len(tokenized_text)*[1] + (512-len(tokenized_text))*[0])
            
            with torch.no_grad():
                last_hidden = self.model(tokens_tensor.unsqueeze(0),attention_mask=attention_mask.unsqueeze(0))[0]
                sent_rep = torch.mean(last_hidden.squeeze(0)[:sent_len],0, False)
                ids.append(sent_rep)
            
        ids = torch.stack(ids, dim=0)
        return ids    

def test(args,source,features):
    
    model_bin = [args.task, args.model_name, args.train_task, args.negative_samples, args.batch_size, args.gradient_accumulation_steps, args.lr, 'train' ]
    model_bin.append(features)
    
    if args.model_name == 'coh_model':
        model = Coh_base()
    else:
        print('Undefined model')
    print(model)    
    
       
       
        
    if torch.cuda.device_count() > 1:
        print('You use %d GPUs'%torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
        if os.path.exists(os.path.join(args.model_path,  '_'.join([str(i) for i in model_bin]))):
            model = torch.load(os.path.join(args.model_path, '_'.join([str(i) for i in model_bin])))
            print('Pretrained model has been loaded')
        else:
            print('Pretrained model does not exist!!!')
            
        #Run the model on a specified GPU and run the operations to multiple GPUs
        torch.cuda.set_device(int(args.cuda))
        model.cuda(int(args.cuda))
        loss_fn= MarginRankingLoss(5.0).cuda(int(args.cuda))
    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        if os.path.exists(os.path.join(args.model_path,  '_'.join([str(i) for i in model_bin]))):
            model = torch.load(os.path.join(args.model_path, '_'.join([str(i) for i in model_bin])))
            print('Pretrained model %s has been loaded'%(os.path.join(args.model_path,  '_'.join([str(i) for i in model_bin]))))
        else:
            print('Pretrained model does not exist!!!')
        model.to(device)
        loss_fn= MarginRankingLoss(5.0)
     
        
    print('The model has {:} different named parameters.\n'.format(len(list(model.named_parameters()))))
    
    if features == 'SBert':
        features = Sentence_features()
    elif features == 'Bert':
        features = Word_features()
    
    model.eval()
    
   
    #anchors = []
    #pos_sens = []
    #neg_sens = []
    
    df_session = pd.DataFrame(columns = ['client','session', 'f_pos','f_neg', 'acc', 'loss'])
    index_session = 0
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
                 acc = 0
                 inferences = []
                 test_loss = 0.0
                 for step, (anchor, pos, neg) in tqdm(enumerate(test_loader)):
                    anchor_ids = features(anchor)
                    pos_ids = features(pos)
                    neg_ids = features(neg)
        
                    #anchors.extend(list(anchor))
                    #pos_sens.extend(list(pos))
                    #neg_sens.extend(list(neg))
        
                    #Load data
                    if torch.cuda.device_count() > 1:
                        anchor_ids = anchor_ids.cuda(non_blocking=True)
                        pos_ids  = pos_ids.cuda(non_blocking=True)
                        neg_ids = neg_ids.cuda(non_blocking=True)
                    else:    
                        anchor_ids = anchor_ids.to(device)
                        pos_ids = pos_ids.to(device)
                        neg_ids = neg_ids.to(device)
     
                    with torch.no_grad():        
                
                        pos_scores = model(anchor_ids, pos_ids)    
                        neg_scores = model(anchor_ids, neg_ids) 
                        
                        loss = loss_fn(pos_scores, neg_scores)
                        test_loss += loss.item()
                        
                        pos_scores = pos_scores.cpu().detach().numpy().flatten().tolist()
                        f_pos += sum(pos_scores)
                        
                        neg_scores = neg_scores.cpu().detach().numpy().flatten().tolist()
                        f_neg += sum(neg_scores)
                        
                        
                        for i in range(len(pos_scores)):
                            if pos_scores[i]>neg_scores[i]:
                                inferences.extend([1])
                            else:
                                inferences.extend([0])
                                
                 f_pos = f_pos/(len(seq) -2)
                 f_neg = f_neg/(len(seq) -2)
                 test_loss = test_loss/(len(seq) -2)  
                 acc = inferences.count(1)/len(inferences)           
            
                 df_session.loc[index_session] = pd.Series({'client':int(client), 'session':int(session), 'f_pos':f_pos, 'f_neg':f_neg, 'acc':acc, 'loss':test_loss})
                 index_session +=1
               
                        
    
          
    
       
    df_session.to_csv('best/'+'_'.join([str(i) for i in model_bin]) + '.csv')
   
    

    
if __name__ == "__main__":
    args = parse_arguments()
    source = 'data/test'
    features = 'Bert' #Can be either SBert or Bert
    
    
  
    USE_CUDA = torch.cuda.is_available()
    test(args,source,features)

