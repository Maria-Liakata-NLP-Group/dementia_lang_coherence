from transformers import  BertForNextSentencePrediction,BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bert(torch.nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.configuration = BertConfig.from_pretrained('bert-large-uncased', output_hidden_states = True, output_attentions = True)
        self.Bert = BertForNextSentencePrediction.from_pretrained('bert-large-uncased')
        #This part is only to train the model from scratch
        #self.Bert =  BertForNextSentencePrediction(self.configuration)
# =============================================================================
#         #Freeze bert layers
#         if self.freeze:
#             for p in self.Bert.parameters():
#                 p.requires_grad = False
# =============================================================================
                
        
    def forward(self, input_ids,attention_masks, type_ids):
        #Feeding the input to BERT model to obtain contextualized representations
        outputs = self.Bert(input_ids = input_ids, attention_mask=attention_masks, type_ids=type_ids)
# =============================================================================
#         #Obtaining the representation of [CLS] head
#         hidden_state = Bert_reps[0]
#         cls_rep = hidden_state[:, 0]
#         #Feeding cls_rep to the fc layer
#         fc_out = F.relu(self.fc(cls_rep))
#         fc_out = F.dropout(fc_out, 0.3)
#         #Feeding fc_out to the classifier layer
#         output = self.classifier(fc_out)
# =============================================================================
        
        return outputs.logits

