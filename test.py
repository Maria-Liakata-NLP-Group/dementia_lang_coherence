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
from transformers import BertModel,BertTokenizer,RobertaTokenizer
from data_loader import Dataload
from torch.utils.data import DataLoader
# Set the seed value all over the place to make this reproducible.



class Extract_feautres(nn.Module):
    def __init__(self,  model):
        super(Extract_feautres, self).__init__()
        self.model = model
        
        if self.model =='BERT_base':
            self.tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.model =='RoBERTa':
            self.tokenizer  = RobertaTokenizer.from_pretrained('roberta-base')
        else:
            print('Undefined model')
    def forward(self, anchors, sentences):
        
        ids = []
        masks = []
        for i,anchor in enumerate(anchors):
            
            sen1 = anchor
            sen2 = sentences[i]
            
            token1 = self.tokenizer.tokenize(sen1)
            token2 = self.tokenizer.tokenize(sen2)
            
            tokens = [self.tokenizer.cls_token] + token1 + [self.tokenizer.sep_token] + token2
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            input_ids = input_ids + [0]*(512-len(input_ids))
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(len(tokens)*[1] + (512-len(tokens))*[0])
            
            ids.append(input_ids)
            masks.append(attention_mask)
        
            
        ids = torch.stack(ids, dim=0)
        masks = torch.stack(masks, dim=0)
        return ids,masks 

class Extract_CNN_feautres(nn.Module):  
    def __init__(self,  model):
        super(Extract_CNN_feautres, self).__init__()
        
        self.tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
    def forward(self, anchors, sentences): 
        
         batch = []
         for i,anchor in enumerate(anchors):
             
             sen1 = anchor
             sen2 = sentences[i]
             
             token1 = self.tokenizer.tokenize(sen1)
             token2 = self.tokenizer.tokenize(sen2)
             
             tokens =  token1 + [self.tokenizer.sep_token] + token2
             input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
             
             segments_ids = [1] * len(tokens)
             
             input_ids = input_ids + [self.tokenizer.pad_token_id]*(512-len(input_ids))
             segments_ids = segments_ids + [self.tokenizer.pad_token_id]*(512-len(segments_ids))
             
             input_ids = torch.tensor(input_ids).unsqueeze(0)
             segments_ids = torch.tensor(segments_ids).unsqueeze(0)
             
             self.model.eval()
             
             with torch.no_grad():
                 encoded_layers = self.model(input_ids,segments_ids)[0]
                 batch.append(encoded_layers)
         batch = torch.stack(batch, dim=0).squeeze(1)    
         
         return batch

def test(args,source):
    
    model_bin = [args.task, args.model_name, args.train_task, args.negative_samples, args.batch_size, args.gradient_accumulation_steps, args.lr, 'train' ]
    
    
    if args.model_name == 'BERT_large':
        model = Bert()
    elif args.model_name == 'RoBERTa':
        model = Roberta()
    elif args.model_name == 'BertCNN':
        model = BertCNN()    
    elif args.model_name == 'BERT_base':
        model = Bert_base()  
    elif args.model_name == 'CNN':
        model = CNN()       
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
        loss_fct = torch.nn.BCELoss().cuda(int(args.cuda))
    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        if os.path.exists(os.path.join(args.model_path,  '_'.join([str(i) for i in model_bin]))):
            model = torch.load(os.path.join(args.model_path, '_'.join([str(i) for i in model_bin])))
            print('Pretrained model %s has been loaded'%(os.path.join(args.model_path,  '_'.join([str(i) for i in model_bin]))))
        else:
            print('Pretrained model does not exist!!!')
        model.to(device)
        loss_fct = torch.nn.BCELoss()
     
        
    print('The model has {:} different named parameters.\n'.format(len(list(model.named_parameters()))))
    
    if args.model_name == 'RoBERTa' or args.model_name == 'BERT_base':
        features = Extract_feautres(args.model_name)
    elif args.model_name == 'CNN':
        features = Extract_CNN_feautres(args.model_name)
    
    df_session = pd.DataFrame(columns = ['client','session', 'f_pos','f_neg', 'acc', 'loss'])
    index_session = 0
   
    model.eval()
    for subdir , dirs, files in os.walk(source): 
        for file in tqdm(files):
            file_path = os.path.join(subdir, file)
            if file.endswith(".csv"):
                print(file)
                client = file.split('.')[0].split('-')[0].split('_')[1]
                session = file.split('.')[0].split('-')[1]
                
                seq = pd.read_csv(os.path.join(source, file))['Turn'].tolist()
                
            
                
                df_data = pd.DataFrame(columns = ['Client','Session', 'Anchor','Sentence', 'Label'])
                index = 0
                
                #Create possitive samples
                for i in range(len(seq) -2):
                    anchor = seq[i].replace('\n', '').replace(',0','').replace(',1','')
                   
                    
                    Sentence =seq[i+1].replace('\n', '').replace(',0','').replace(',1','')
                    df_data.loc[index] = pd.Series({'Client':int(client), 'Session':int(session), 'Anchor':anchor, 'Sentence':Sentence, 'Label':1})
                    index+=1
                    
                #Create negative samples 
                for i in range(len(seq)):
                    acnhor = seq[i].replace('\n', '').replace(',0','').replace(',1','')
                    for j in range(len(seq)):
                        if  j>i+1:
                            Sentence = seq[j].replace('\n', '').replace(',0','').replace(',1','')
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
                    
                    if args.model_name == 'RoBERTa' or args.model_name == 'BERT_base':
                        pos_ids, pos_masks = features(anchor,pos)
                        neg_ids, neg_masks = features(anchor,neg)
                        input_ids = torch.cat((pos_ids,neg_ids), dim=0)
                        attention_masks = torch.cat((pos_masks,neg_masks), dim=0)
                        labels = torch.tensor(len(pos)*[1] + len(neg)*[0]).to(torch.float32)
                        #Load data
                        if torch.cuda.device_count() > 1:
                            input_ids = input_ids.cuda(non_blocking=True)
                            attention_masks  = attention_masks.cuda(non_blocking=True)
                            labels = labels.cuda(non_blocking=True)
                        else:    
                            input_ids = input_ids.to(device)
                            attention_masks  = attention_masks.to(device)
                            labels = labels.to(device)
                        
                        with torch.no_grad():  
                            logits = torch.squeeze(model(input_ids,attention_masks))    
                        
                    elif args.model_name == 'CNN':  
                        pos_ids = features(anchor,pos)
                        neg_ids = features(anchor,neg)
                        input_ids = torch.cat((pos_ids,neg_ids), dim=0)
                        labels = torch.tensor(len(pos)*[1] + len(neg)*[0]).to(torch.float32)
                        #Load data
                        if torch.cuda.device_count() > 1:
                            input_ids = input_ids.cuda(non_blocking=True)
                            labels = labels.cuda(non_blocking=True)
                        else:    
                            input_ids = input_ids.to(device)
                            labels = labels.to(device)
                        
                        with torch.no_grad():  
                            logits = torch.squeeze(model(input_ids))  
                        
                    loss = loss_fct(logits, labels)
                    test_loss += loss.item()
                        
                    pos_scores = logits[:len(pos)]
                    pos_scores = pos_scores[0].cpu().detach().numpy().flatten().tolist()
                    f_pos += sum(pos_scores)
                        
                    neg_scores = logits[len(pos):]
                    neg_scores = neg_scores[0].cpu().detach().numpy().flatten().tolist()
                    f_neg += sum(neg_scores)
                        
                    pos_case.append(pos_scores)
                    neg_case.append(neg_scores)
                        
                         
                    for i in range(len(pos_scores)):
                        if pos_scores[i]>neg_scores[i]:
                            inferences.extend([1])
                        else:
                            inferences.extend([0])
                
                f_pos = f_pos/(len(seq) -2)     
                print(f_pos)
                f_neg = f_neg/(len(seq) -2)    
                test_loss = test_loss/(len(seq) -2)   
                acc = inferences.count(1)/len(inferences)                     
                df_session.loc[index_session] = pd.Series({'client':int(client), 'session':int(session), 'f_pos':f_pos, 'f_neg':f_neg, 'acc':acc, 'loss':test_loss})
                index_session +=1
                
                #print('\n')
                #print(pos_case)
                #print('\n')
                #print(neg_case)
    
    df_session.to_csv('best/'+'_'.join([str(i) for i in model_bin]) + '.csv')
    
if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    source = 'data/test/'
  
    USE_CUDA = torch.cuda.is_available()
    test(args,source)
