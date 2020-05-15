import torch.nn as nn

class RN101_newtop(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base = base_model
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, input):
        x = self.base(input)
        bs, c, h, w = x.shape
        x = x.reshape(bs,-1);  #need to reshape for linear layer 
        x = self.fc(x)
        return x
