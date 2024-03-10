import os
import random
import numpy as np
import torch
import torch.nn as nn
from argument_parser import parse_arguments
from data_loader import Dataload
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
from datasets import load_metric
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


class Extract_feautres_labels(nn.Module):
    def __init__(self,  model):
        super(Extract_feautres_labels, self).__init__()
        self.model = model
        
        self.tokenizer  = T5Tokenizer.from_pretrained(self.model)
    def forward(self, sentences):
        
        ids = []
        masks = []
        for i,sentence in enumerate(sentences):
            
            
            _input = self.tokenizer(sentence, truncation=True, max_length=512, padding='max_length',return_tensors="pt")
            
            ids.append(_input['input_ids'])
            masks.append(_input['attention_mask'])
            
        ids = torch.stack(ids, dim=0).squeeze(1)
        masks = torch.stack(masks, dim=0).squeeze(1)
        
        return ids,masks    



def test(args):
    

    model_bin = [args.task, args.model_name, args.train_task, args.negative_samples, args.batch_size, args.gradient_accumulation_steps, args.lr, 'train' ]
   
    model_name = '_'.join([str(i) for i in model_bin])
    
    if 'gpt' in args.model_name:
        None
    elif 't5' in args.model_name:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        print('Undefined model')
    
       
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
    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        if os.path.exists(os.path.join(args.model_path,  '_'.join([str(i) for i in model_bin]))):
            model = torch.load(os.path.join(args.model_path, '_'.join([str(i) for i in model_bin])))
            print('Pretrained model %s has been loaded'%(os.path.join(args.model_path,  '_'.join([str(i) for i in model_bin]))))
        else:
            print('Pretrained model does not exist!!!')
        model.to(device)
    
    print('The model has {:} different named parameters.\n'.format(len(list(model.named_parameters())))) 
   
    
    
    features_label = Extract_feautres_labels(args.model_name) 
    
    
    
    # Reset the total loss for this epoch.
    df_session = pd.DataFrame(columns = ['client','session', 'f_pos','f_neg', 'acc', 'loss'])
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
                    input_ids, attention_masks = features_label(anchor)
                    pos_labels, _ = features_label(pos)
                    neg_labels, _ = features_label(neg)
    
    
                    #Load data
                    if torch.cuda.device_count() > 1:
                        input_ids = input_ids.cuda(non_blocking=True)
                        attention_masks  = attention_masks.cuda(non_blocking=True)
                        pos_labels = pos_labels.cuda(non_blocking=True)
                        neg_labels = neg_labels.cuda(non_blocking=True)
                    else:    
                        input_ids = input_ids.to(device)
                        attention_masks  = attention_masks.to(device)
                        pos_labels = pos_labels.to(device)
                        neg_labels = neg_labels.to(device)
                        
                    with torch.no_grad():     
                        outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=pos_labels)
                        log_likelihood = outputs.loss
                        test_loss += outputs.loss.item()
                        pos_scores = [torch.exp(log_likelihood)]
                        pos_scores = pos_scores[0].cpu().detach().numpy().flatten().tolist()
                        pos_scores = [1 - score for score in pos_scores]
                        f_pos += sum(pos_scores)
                        pos_case.append(pos_scores[0])
                        
                        outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=neg_labels)
                        log_likelihood = outputs.loss
                        test_loss += outputs.loss.item()
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
                test_loss = test_loss/(len(seq) -2)  
                df_session.loc[index_session] = pd.Series({'client':int(client), 'session':int(session), 'f_pos':f_pos, 'f_neg':f_neg, 'acc':acc, 'loss':test_loss})
                index_session +=1
                
    df_session.to_csv('best/'+'_'.join([str(i) for i in model_bin]) + '.csv')
    
if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    source = 'data/test'
    USE_CUDA = torch.cuda.is_available()
    test(args)       
