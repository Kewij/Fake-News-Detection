import torch.nn as nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, nb_classes):
        super(BERTClassifier, self).__init__()

        self.nb_classes = nb_classes
        self.pt_model = BertModel.from_pretrained('bert-base-uncased')
        #for param in self.pt_model.parameters():
        #    param.requires_grad = False
        self.fc = nn.Linear(self.pt_model.config.hidden_size, self.nb_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.pt_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(pooled_output)
        return logits