# Imports
import math
import copy
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from colorama import Fore, Back, Style, init
init(autoreset = True)

# Hyper-Parameters & Fake Tensors to test
batchSize = 2
seqLength = 5
mDim = 8
nHeads = 2
ffDim = 3
Q = torch.randn(batchSize, seqLength, mDim)
K = torch.randn(batchSize, seqLength, mDim)
V = torch.randn(batchSize, seqLength, mDim)
x = torch.randn(batchSize, seqLength, mDim)
print(Fore.YELLOW + 'Batch Size:', Fore.CYAN + str(batchSize))
print(Fore.YELLOW + 'Sequence Length:', Fore.CYAN + str(seqLength))
print(Fore.YELLOW + 'Model Dimensions:', Fore.CYAN + str(mDim))
print(Fore.YELLOW + 'Number of Heads:', Fore.CYAN + str(nHeads))
print(Fore.YELLOW + 'Number of FFDim:', Fore.CYAN + str(ffDim))
print(Fore.BLUE + 'Shape of K, Q, V Tensors:', Fore.RED + str(Q.shape))

# Positional Embedding 
class PerPositionFeedForward(nn.Module):
    def __init__(self, mDim: int, ffDim: int) -> None:
        '''
        We use this feed forward network instead of simple FF throughout the whole code
        `mDim:` dimensionality of the model's input & output
        `ffDim:` dimensionality of inner feed-forward layer
        `IMPORTANT:` this feed-forward network is applied to each `position`
        separately and identically
        '''
        super(PerPositionFeedForward, self).__init__()
        self.fc1 = nn.Linear(mDim, ffDim)
        self.fc2 = nn.Linear(ffDim, mDim)
        self.relu = nn.ReLU()

    def forward(self, X: Tensor) -> Tensor:
        '''
                    ▲
                   (+) <--------  Positional Embedding
                    ▲
            ┌──────────────┐ 
            │ feed-forward │ 
            └──────────────┘
                    ▲
                  inputs
        '''
        string = '[PerPositionFeedForward STARTED!]'
        print(Fore.YELLOW + string, (80 - len(string)) * '-')
        print(Fore.BLUE + 'X Shape:', Fore.RED + str(X.shape))
        out = self.fc2(self.relu(self.fc1(X)))
        print(Fore.BLUE + 'Output Shape:', Fore.RED + str(out.shape))
        print(out)
        return out

# Tests
# ppff = PerPositionFeedForward(mDim, ffDim)
# ppff.forward(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, mDim: int, maxSeqLen: int) -> None:
        '''
        Positional Encoding is used to `inject` the position information of each token in 
        the input sequence. It uses `sine` and `cosine` functions of different frequencies
        to generate the positional encoding.
        `mDim:` dimensionality of the model's input & output
        `maxSeqLen:` maximum sequence length
        '''
        super(PositionalEmbedding, self).__init__()
        string = '[PositionalEncoding]'
        print(Fore.YELLOW + string, (80 - len(string)) * '-')
        pe = torch.zeros(maxSeqLen, mDim)
        position = torch.arange(0, maxSeqLen, dtype = torch.float).unsqueeze(1)
        print(Fore.BLUE + 'Positions:')
        print(position)
        # divTerm[i] = exp(-ln(10000) * i / d_model),
        divisonTerm = torch.exp(torch.arange(0, mDim, 2).float() * -(math.log(10000.0) / mDim))
        pe[:, 0::2] = torch.sin(position * divisonTerm)
        pe[:, 1::2] = torch.cos(position * divisonTerm)
        print(Fore.BLUE + 'New Positions:')
        print(pe)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, X: Tensor) -> Tensor:
        '''
                    ▲
                   (+) <--------  Positional Embedding
                    ▲

        `Important`: We're not only Embedding using sin & cos but actually sending X
        along with the positional encoding
        '''
        print(Fore.BLUE + 'X Shape:', Fore.RED + str(X.shape))
        out = X + self.pe[:, :X.size(1)]
        print(Fore.BLUE + 'Output Shape:', Fore.RED + str(out.shape))
        print(out)
        return out

# Tests
# pe = PositionalEmbedding(mDim, 10)
# pe.forward(x)

# Multi-Head Attention
class MHA(nn.Module):
    def __init__(self, mDim: int, nHeads: int) -> None:
        '''
        Mult-head Attention Block is the `MOST` important block of the transformers
        `mDim:` dimensionality of the model's input & output
        `nHeads:` number of heads 
        `Warning: mDim must be devisible by nHeads`
        '''
        super(MHA, self).__init__()
        assert mDim % nHeads == 0, 'mDim must be devisible by nHeads'
        # Initialize dimentions
        self.mDim = mDim
        self.nHeads = nHeads
        self.dimKQV = mDim // nHeads
        # Linear layers
        self.wQ = nn.Linear(mDim, mDim)
        self.wK = nn.Linear(mDim, mDim)
        self.wV = nn.Linear(mDim, mDim)
        self.wO = nn.Linear(mDim, mDim)

    def scaledDotProductAttention(self, K: Tensor, Q: Tensor, V: Tensor, 
                                  mask: Tensor | None = None) -> Tensor:
        ''' `Attention(Q, K, V)` = Softmax((Q · K^T) / √d_k) · V
        `Q:` Query, `K:` Key, `V:` Value
        '''
        string = '[Scaled Dot Product Attention]'
        print(Fore.CYAN + string, (80 - len(string)) * '-')
        attentionScores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dimKQV)
        print(Fore.BLUE + 'Attention Score Shape:', Fore.RED + str(attentionScores.shape))

        attentionProbs = torch.softmax(attentionScores, dim = -1)
        print(Fore.BLUE + 'Attention Probabilty (Softmax) Shape:', 
              Fore.RED + str(attentionProbs.shape))

        if mask is not None:
            attentionScores = attentionScores.masked_fill(mask == 0, -torch.inf)

        output = torch.matmul(attentionProbs, V)
        print(Fore.YELLOW + 'Scaled Dot Product Output Shape:', 
              Fore.CYAN + str(output.shape))

        return output

    def splitHeads(self, X: Tensor, who: str) -> Tensor:
        '''Reshapes the `input` to have `number of heads` as a dimension'''
        string = f'[Spliting Heads for {who}]'
        print(Fore.CYAN + string, (80 - len(string)) * '-')
        print(Fore.BLUE + 'Input X Shape:', Fore.RED + str(X.shape))
        batchSize, seqLength, _ = X.size()
        reshapedX = x.view(batchSize, seqLength, self.nHeads, self.dimKQV).transpose(1, 2)
        print(Fore.BLUE + 'Reshped X Shape:', Fore.RED + str(reshapedX.shape))
        return reshapedX

    def combineHeads(self, X: Tensor) -> Tensor:
        '''Combines the multiple heads back to original shape'''
        string = '[Combine Heads]'
        print(Fore.CYAN + string, (80 - len(string)) * '-')
        batchSize, _, seqLength, dimKQV = X.size()
        reshapedX = x.transpose(1, 2).contiguous().view(batchSize, seqLength, self.mDim)
        print(Fore.BLUE + 'Concatenated Output Shape:', Fore.RED + str(reshapedX.shape))
        return reshapedX

    def forward(self, K:Tensor, Q: Tensor, V: Tensor, mask: Tensor = None) -> None:
        '''
              Linear ──►  Decoder
                ▲
              Concat
                ▲
        ┌──────────────┐ 
        │Scaled dot att│ nHeads
        └──────────────┘
          ▲    ▲    ▲
         Lin  Lin  Lin
          ▲    ▲    ▲
            
          V    K    Q
        '''
        string = '[FORWARD CALL]'
        print(Fore.YELLOW + string, (80 - len(string)) * '-')
        Q = self.splitHeads(self.wQ(Q), 'Q') # Q -> Linear(wQ) -> Q -> Att ...
        K = self.splitHeads(self.wK(K), 'K') # K -> Linear(wK) -> K -> Att ...
        V = self.splitHeads(self.wV(V), 'V') # V -> Linear(wV) -> V -> Att ...

        attentionOutput = self.scaledDotProductAttention(K, Q, V, mask)
        output = self.wO(self.combineHeads(attentionOutput))
        # string = '[MHA OUTPUT]'
        # print(Fore.YELLOW + string, (80 - len(string)) * '-')
        # print(output)
        return output

# Tests
# mha = MHA(mDim, nHeads)
# output = mha.forward(K, Q, V)

class EncoderLayer(nn.Module):
    def __init__(self, mDim: int, nHeads: int, ffDim: int, dropout: float) -> None:
        '''
                  ...  
                   ▲ 
        ┌──────────────────────┐
        │      Add & Norm      │
        └──────────────────────┘
                   ▲ 
        ┌──────────────────────┐
        │Position-Wise Feed FW │
        └──────────────────────┘
                   ▲ 
        ┌──────────────────────┐
        │      Add & Norm      │
        └──────────────────────┘
                   ▲ 
        ┌──────────────────────┐
        │  Multi-Head Attention│ <--- Q, K, V
        └──────────────────────┘
        '''
        super(EncoderLayer).__init__()
        self.selfAttention = MHA(mDim, nHeads)
        self.norm1 = nn.LayerNorm(mDim)
        self.feedForward = PerPositionFeedForward(mDim, ffDim)
        self.norm2 = nn.LayerNorm(mDim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor, mask: Tensor | None) -> Tensor:
        attnOutput = self.selfAttention(X)
        X = self.norm1(X + self.dropout(attnOutput)) # We do this beause of residual connections
        ffOutput = self.feedForward(X)
        X = self.norm2(X + self.dropout(ffOutput))
        return X
