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
Q = torch.randn(batchSize, seqLength, mDim)
K = torch.randn(batchSize, seqLength, mDim)
V = torch.randn(batchSize, seqLength, mDim)
x = torch.randn(batchSize, seqLength, mDim)
print(Fore.YELLOW + 'Batch Size:', Fore.CYAN + str(batchSize))
print(Fore.YELLOW + 'Sequence Length:', Fore.CYAN + str(seqLength))
print(Fore.YELLOW + 'Model Dimensions:', Fore.CYAN + str(mDim))
print(Fore.YELLOW + 'Number of Heads:', Fore.CYAN + str(nHeads))
print(Fore.BLUE + 'Shape of K, Q, V Tensors:', Fore.RED + str(Q.shape))

# TODO! Positional Embedding + Linear

# Multi-Head Attention
class MHA(nn.Module):
    def __init__(self, mDim: int, nHeads: int) -> None:
        '''
        Mult-head Attention Block is the `MOST` important block of the transformers
        `mDim:` model dimension 
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

mha = MHA(mDim, nHeads)
output = mha.forward(K, Q, V)
