import torch
from torch import nn
    
class Embedding(nn.Module):
    def __init__(self, dim=4, voc=10):
        super().__init__()
        self.dim = dim
        self.voc = voc
        self.table = nn.Embedding(self.voc, self.dim)
    
    def forward(self, x): # B,S (1,6)  -> B,S,H (1,6,4)
        return self.table(x)
        

if __name__ == '__main__':
    data = torch.ones(2, 3, dtype=torch.long)
    print("data", data)
    table = Embedding()
    print("table", table.table.weight.shape, table.table.weight)
    embedding = table(data)   
    print("embedding", embedding.shape, embedding) 
