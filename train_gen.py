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



class Extract_feautres(nn.Module):
    def __init__(self,  model):
        super(Extract_feautres, self).__init__()
        self.model = model
        
        self.tokenizer  = T5Tokenizer.from_pretrained(self.model)
    def forward(self, anchors, sentences):
        
        ids = []
        masks = []
        for i,anchor in enumerate(anchors):
            
            sen1 = anchor
            sen2 = sentences[i]
            
            pair = sen1  + sen2
            
            _input = self.tokenizer(pair, truncation=True, max_length=512, padding='max_length',return_tensors="pt")
            
            ids.append(_input['input_ids'])
            masks.append(_input['attention_mask'])
            
        ids = torch.stack(ids, dim=0).squeeze(1)
        masks = torch.stack(masks, dim=0).squeeze(1)
        
        return ids,masks
    
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






def compute_blue(predictions, labels, tokenizer, train=False):
    ''' 
    BLEU is precision focused
    calculate pair wise bleu score. uses nltk implementation
    Args:
        references : a list of reference sentences 
        candidates : a list of candidate(generated) sentences
    Returns:
        bleu score(float)
    '''
    blue = 0
    for i in range (0,labels.size(0)):
        label = labels[i,:]
        prediction = predictions[i,:]
        
        ref = []
        gen = []
    
        if train==True:
            prediction = torch.argmax(prediction, dim=-1)
        generated_text = prediction.tolist()
        label = label.tolist()
        
        ref.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(label,skip_special_tokens=True)))
        gen.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(generated_text,skip_special_tokens=True)))
        
        ref_blue = []
        gen_blue = []
        for l in gen:
            gen_blue.append(l.split())
        for i,l in enumerate(ref):
            ref_blue.append([l.split()])
        cc = SmoothingFunction()
        score_bleu = corpus_bleu(ref_blue, gen_blue, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method4)
        blue += score_bleu
       
        
        
    batch_blue = blue/labels.size(0)
    return batch_blue

def compute_rouge(predictions, labels, tokenizer, train=False):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge = 0.0    
    for i in range (0,labels.size(0)):
        label = labels[i,:]
        prediction = predictions[i,:]
        
        ref = []
        gen = []
        
        if train==True:
            prediction = torch.argmax(prediction, dim=-1)
        generated_text = prediction.tolist()
        label = label.tolist()
        
        ref.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(label,skip_special_tokens=True)))
        gen.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(generated_text,skip_special_tokens=True)))
        
        scores = scorer.score(ref[0],gen[0])
        rouge += scores['rouge1'][2] #fmeasure
        
    batch_rouge = rouge/labels.size(0)    
    return batch_rouge
        

def train(args):
    
    
    
    model_bin = [args.task, args.model_name, args.train_task, args.negative_samples, args.batch_size, args.gradient_accumulation_steps, args.lr, 'train' ]
    df = pd.DataFrame(columns = ['Epoch','Tr Loss', 'Val Loss', 'Tr Acc', 'Val Acc', 'lr'])
    
    if 'gpt' in args.model_name:
        None
    elif 't5' in args.model_name:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        print('Undefined model')
        
        
    if torch.cuda.device_count() > 1:
        print('You use %d GPUs'%torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0])
        #Run the model on a specified GPU and run the operations to multiple GPUs
        torch.cuda.set_device(int(args.cuda))
        model.cuda(int(args.cuda))
    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        model.to(device)
        
        
       
    print('The model has {:} different named parameters.\n'.format(len(list(model.named_parameters())))) 
   
        

    optimizer = Adafactor(model.parameters(),lr=1e-3,
                  eps=(1e-30, 1e-3),
                  clip_threshold=1.0,
                  decay_rate=-0.8,
                  beta1=None,
                  weight_decay=0.0,
                  relative_step=False,
                  scale_parameter=False,
                  warmup_init=False)
        
        
    features = Extract_feautres(args.model_name)
    features_label = Extract_feautres_labels(args.model_name)
    
    model.zero_grad()
    best_loss = 1000
    trigger_times = 0
    
    #train_iterator = tqdm.notebook.tnrange(int(args.epochs), desc="Epoch")
    for epoch in range(0, args.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.epochs))
    
        #Load data
        df_train = pd.read_csv('data/disc_train_same_.csv')
        train_data = Dataload(df_train)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,shuffle=True,drop_last=True)
        
        
        df_val = pd.read_csv('data/disc_val_same_.csv')
        val_data = Dataload(df_val)
        val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size,shuffle=False,drop_last=True)
        
        
        print('Training...')
        model.train()
        # Reset the total loss for this epoch.
        # Reset the total loss for this epoch.
        tr_loss = 0.0
        tr_global_steps = 0
        
        for step, (anchor, pos, neg) in tqdm(enumerate(train_loader)):
            input_ids, attention_masks = features(anchor,pos)
            labels, _ = features_label(pos)
            
            #Load data
            if torch.cuda.device_count() > 1:
                input_ids = input_ids.cuda(non_blocking=True)
                attention_masks  = attention_masks.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            else:    
                input_ids = input_ids.to(device)
                attention_masks  = attention_masks.to(device)
                labels = labels.to(device)
           
                
            if 'gpt' in args.model_name:
                None
            elif 't5' in args.model_name: 
                # Forward propogation
                outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss
        
            
            # accumulate train loss
            loss = loss/args.gradient_accumulation_steps
            tr_loss += loss.item()
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if (step+1)% args.gradient_accumulation_steps==0:
                optimizer.step()
                model.zero_grad()
                tr_global_steps +=1
                
                
                
            if (tr_global_steps+1)%args.statistic_step==0:
                print('\n')
                print('Train loss per %d steps:%0.3f'%(tr_global_steps+1,(tr_loss/(tr_global_steps+1))))
              
               
        print('\n')         
        print("")
        print('Training resuts')  
       
        # Calculate the average loss over all of the batches
        print('Train loss per %d steps:%0.3f'%(tr_global_steps+1,(tr_loss/(tr_global_steps+1))))
        

        print("")
        print("Running Validation...")   
        
      
        model.eval()
        val_loss = 0.0
        val_global_steps = 0
        # Evaluate data for one epoch
        
        for step, (anchor, pos, neg) in tqdm(enumerate(val_loader)):
            input_ids, attention_masks = features_label(anchor)
            labels, _ = features_label(pos)
            
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
                    if 'gpt' in args.model_name:
                        None
                    elif 't5' in args.model_name:    
                        outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
                        loss = outputs.loss
                        
# =============================================================================
#                    model.generate(tokenized_text,
#                                     num_beams=4,
#                                     no_repeat_ngram_size=2,
#                                     min_length=30,
#                                     max_length=100,
#                                     early_stopping=True)
# =============================================================================
                    val_loss +=loss.item()
                    
                    
                    if (step+1)% args.gradient_accumulation_steps==0:
                        val_global_steps+=1
                        
                    if (val_global_steps+1)%args.statistic_step==0:  
                        print('\n')
                        print('Val loss per %d steps:%0.3f'%(val_global_steps+1,(val_loss/(val_global_steps+1))))
                      
                        
        
        print("")        
        print('\n')
        print('Validation resuts')
        print('Val loss per %d steps:%0.3f'%(val_global_steps+1,(val_loss/(val_global_steps+1))))

        
          
        if val_loss/(val_global_steps+1)<best_loss:
            trigger_times = 0
            best_loss = val_loss/(val_global_steps+1)
            print('Saving trained model...')
            model_file = os.path.join(args.model_path, '_'.join([str(i) for i in model_bin]))
            torch.save(model,model_file)
        else:
            trigger_times += 1
            if trigger_times>= args.patience:
                print('Early Stopping!')
                break 
    print("")
    #df.to_csv(os.path.join(args.model_path,'_'.join([str(i) for i in model_bin]))+'.csv')
    print("Training complete!")  


    
if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    
    USE_CUDA = torch.cuda.is_available()
    train(args)
    
   
