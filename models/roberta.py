from transformers import  RobertaModel, RobertaConfig,RobertaForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F


class Roberta(torch.nn.Module):
    def __init__(self):
        super(Roberta, self).__init__()
        self.configuration = RobertaConfig.from_pretrained('roberta-base')
        self.Roberta = RobertaForSequenceClassification.from_pretrained("roberta-base")
        self.num_classes = 1
        #This part is only to train the model from scratch
        #self.Bert =  BertForNextSentencePrediction(self.configuration)
# =============================================================================
#         #Freeze bert layers
#         if self.freeze:
#             for p in self.Bert.parameters():
#                 p.requires_grad = False
# =============================================================================
        self.classifier = nn.Linear(self.configuration.hidden_size, self.num_classes)        
                  
        
    def forward(self, input_ids,attention_masks):
        #Feeding the input to BERT model to obtain contextualized representations
        roperta_out = self.Roberta(input_ids = input_ids, attention_mask=attention_masks)
        #Obtaining the representation of [CLS] head
        hidden_state = roperta_out[0]
        cls_rep = hidden_state[:, 0]
        logits = self.classifier(cls_rep)
        logits = torch.sigmoid(logits)
        
        return logits

