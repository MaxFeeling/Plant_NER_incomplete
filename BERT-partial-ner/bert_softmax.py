import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel,BertModel
from torch.nn import CrossEntropyLoss

class BertSoftMax(BertPreTrainedModel):
    def __init__(self,cfig):
        super(BertSoftMax,self).__init__(cfig)

        self.device = cfig.device
        self.num_labels = 7
        self.bert = BertModel(cfig)
        self.dropout = nn.Dropout(cfig.hidden_dropout_prob)
        self.classifier = nn.Linear(cfig.hidden_size, 7)
        self.init_weights()

    def forward(self,input_ids, input_seq_lens=None, annotation_mask=None, labels=None,
                attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1,self.num_labels),labels.view(-1))
            outputs = (loss,) + (logits,)+outputs[2:]
        else:
            outputs = (logits,) + outputs[2:]

        return outputs