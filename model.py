import torch
from torch import nn
import math
    
DEFAULT_DIM = 4
DEFAULT_VOC = 10
DEFAULT_WINDOW = 8
DEFAULT_LAYER = 3
DEFAULT_HEAD = 2
class Embedding(nn.Module):
    def __init__(self, dim=DEFAULT_DIM, voc=DEFAULT_VOC):
        super().__init__()
        self.dim = dim
        self.voc = voc
        self.table = nn.Embedding(self.voc, self.dim)
    
    def forward(self, x): # B,S (1,6)  -> B,S,H (1,6,4)
        return self.table(x)

# multi head attention
class MHAttention(nn.Module):
    def __init__(self, dim=DEFAULT_DIM, voc=DEFAULT_VOC, head=DEFAULT_HEAD):
        super().__init__()
        self.dim = dim
        self.voc=voc
        self.head=head
        self.kqv=nn.Linear(self.dim, 3*self.dim)
        self.out=nn.Linear(self.dim, self.dim)
    def forward(self, x): # bsh -> bsh
        # multi head, we split hidden dimension by num of head
        # get qkv -> bsh
        b,s,h = x.shape
        q,k,v=self.kqv(x).chunk(3, dim=-1) #bsh
        # instead of do qk matul, we need make it bsnd (n*d=h)  and then convert to bnsd
        d = self.dim//self.head # dimension per head
        q = q.view(b,s,self.head,-1).permute(0,2,1,3)
        k = k.view(b,s,self.head,-1).permute(0,2,1,3)
        v = v.view(b,s,self.head,-1).permute(0,2,1,3)
        # attention score, it should still feel like bsh except we added a new dim
        qk = q@k.transpose(-2,-1)/math.sqrt(d) # d is smaller than dim
        qk = nn.functional.softmax(qk, dim=-1)
        qkv = (qk@v).permute(0,2,1,3).reshape(b,s,-1) # swich back to bshd and then bsh
        return self.out(qkv)
        
class Attention(nn.Module):
    def __init__(self, dim=DEFAULT_DIM, voc=DEFAULT_VOC):
        super().__init__()
        self.dim = dim
        self.voc=voc
        self.kqv=nn.Linear(self.dim, 3*self.dim)
        self.out=nn.Linear(self.dim, self.dim)
    
    def forward(self, x): # bsh -> bsh
        q,k, v = self.kqv(x).chunk(3, dim=-1) # b*s*3h -> bsh, bsh, bsh
        print("q", q.shape, q[1])
        # qk will be significant larger than q, so we have to divid by dim (sqrt)
        qk = q@k.transpose(-2,-1)/math.sqrt(self.dim) # bsh @ bhs -> bss
        print("qk", qk)
        # do softmax (sum of each row is one)
        qk = nn.functional.softmax(qk, dim=-1)
        print("qk after softmax", qk)
        return self.out(qk@v)
        
class MLP(nn.Module):
    def __init__(self, dim=DEFAULT_DIM, voc=DEFAULT_VOC):
        super().__init__()
        self.dim = dim
        self.voc = voc
        self.mlp1 = nn.Linear(self.dim, self.dim*2)
        self.mlp2 = nn.Linear(self.dim*2, self.dim)
    
    def forward(self, x):# bsh
        x = self.mlp1(x)
        x = self.mlp2(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, att, mlp):
        super().__init__()
        self.att = att
        self.mlp = mlp
    
    def forward(self, x):
        x = self.att(x)
        return self.mlp(x)

class Transformer(nn.Module):
    def __init__(self, dim=DEFAULT_DIM, voc=DEFAULT_VOC, window=DEFAULT_WINDOW, num_layer=DEFAULT_LAYER):
        super().__init__()
        self.embedding = Embedding()
        self.attention = nn.Sequential()
        self.lm_head = nn.Linear(dim, voc)
        for i in range(num_layer):
            att = Attention()
            mlp = MLP()
            att_mlp = AttentionBlock(att, mlp)
            self.attention.append(att_mlp)
    
    
    def forward(self, x):
        x = self.embedding(x) # BS -> BSH
        # BSH->BSH
        for att in self.attention:
            x = att(x) 
        # return logits?
        logits = self.lm_head(x)
        print("logits", logits.shape)
        return logits

if __name__ == '__main__':
    data = torch.randint(0, 10, (2, 3))
    print("data", data)
    # transformer = Transformer()
    # x = transformer(data)
    # print("x", x)
    table = Embedding()
    print("table", table.table.weight.shape, table.table.weight)
    embedding = table(data)   
    print("embedding", embedding.shape, embedding) 
    
    attention = MHAttention()
    x = attention(embedding)
    # 
    # mlp = MLP()
    # m = mlp(x)
    # print("after mlp", m)
    