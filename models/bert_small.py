from transformers import  BertModel, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bert_base(torch.nn.Module):
    def __init__(self):
        super(Bert_base, self).__init__()
        self.configuration = BertModel.from_pretrained('bert-base-uncased')
        self.Bert = BertModel.from_pretrained('bert-base-uncased')
        self.num_classes = 1
        #This part is only to train the model from scratch
        #self.Bert =  BertForNextSentencePrediction(self.configuration)
# =============================================================================
#         #Freeze bert layers
#         if self.freeze:
#             for p in self.Bert.parameters():
#                 p.requires_grad = False
# =============================================================================
        self.classifier = nn.Linear(self.configuration.pooler.dense.out_features, self.num_classes)        
                
        
    def forward(self, input_ids,attention_masks):
        #Feeding the input to BERT model to obtain contextualized representations
        Bert_reps = self.Bert(input_ids = input_ids, attention_mask=attention_masks)

         #Obtaining the representation of [CLS] head
        hidden_state = Bert_reps[0]
        cls_rep = hidden_state[:, 0]
#         #Feeding cls_rep to the fc layer
#        fc_out = F.relu(self.fc(cls_rep))
#         fc_out = F.dropout(fc_out, 0.3)
#         #Feeding fc_out to the classifier layer
        logits = self.classifier(cls_rep)
        
        logits = torch.sigmoid(logits)
# =============================================================================
        
        return logits

