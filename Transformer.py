# Imports
import math
import copy
import torch
import pandas as pd
import torch.nn as nn
from utils import runEpoch
import torch.optim as optim
import torch.utils.data as data
from torch.functional import Tensor
from colorama import Fore, Back, Style, init
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
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
        batchSize, seqLength, _ = X.size()
        reshapedX = x.view(batchSize, seqLength, self.nHeads, self.dimKQV).transpose(1, 2)
        string = f'[Spliting Heads for {who}]'
        print(Fore.MAGENTA + string, (80 - len(string)) * '-')
        input = Fore.CYAN + str(list(X.shape))
        reXStr = Fore.CYAN + str(list(reshapedX.shape))
        outStr = f'''
            {reXStr}
                 ▲
            {who} : {input}
        '''
        print(outStr)
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
        super(EncoderLayer, self).__init__()
        self.selfAttention = MHA(mDim, nHeads)
        self.norm1 = nn.LayerNorm(mDim)
        self.feedForward = PerPositionFeedForward(mDim, ffDim)
        self.norm2 = nn.LayerNorm(mDim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor, mask: Tensor | None = None) -> Tensor:
        string = '[ENCODER LAYER FORWARD]' 
        print(Fore.YELLOW + string, (80 - len(string)) * '-')
        print(Fore.BLUE + 'Input X Shape:', Fore.RED + str(X.shape))
        attnOutput = self.selfAttention(X, X, X, mask)
        X = self.norm1(X + self.dropout(attnOutput)) # We do this beause of residual connections
        ffOutput = self.feedForward(X)
        X = self.norm2(X + self.dropout(ffOutput))
        return X

# Tests
enc = EncoderLayer(mDim, nHeads, ffDim, 0.2)
output = enc.forward(x) 
print(output)

# Decoder Layer
class DecoderLayer(nn.Module):
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
            │   Cross-Attention    │ <--- K, V (from encoder)
            └──────────────────────┘
                       ▲ 
            ┌──────────────────────┐
            │      Add & Norm      │
            └──────────────────────┘
                       ▲ 
            ┌──────────────────────┐
            │Masked Self-Attention │ <--- Q, K, V
            └──────────────────────┘
        '''
        super(DecoderLayer, self).__init__()
        self.selfAttention = MHA(mDim, nHeads)
        self.norm1 = nn.LayerNorm(mDim)
        self.crossAttention = MHA(mDim, nHeads)
        self.norm2 = nn.LayerNorm(mDim)
        self.feedForward = PerPositionFeedForward(mDim, ffDim)
        self.norm3 = nn.LayerNorm(mDim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor, encOut: Tensor | None = None,
                tgtMask: Tensor | None = None,
                srcMask: Tensor | None = None) -> Tensor:
        attnOutput = self.selfAttention(X, X, X, tgtMask)
        X = self.norm1(X + self.dropout(attnOutput))
        crossAttnOut = self.crossAttention(X, )
        X = self.norm2(X + self.dropout(crossAttnOut))
        ffOutput = self.feedForward(X)
        X = self.norm3(X + self.dropout(ffOutput))
        return X

# Transformers... Assemble!
class Transformer(nn.Module):
    def __init__(self, mDim: int, nHeads: int, ffDim: int, dropout: float, maxSeqLen: int,
                 srcVocabSize: int, tgtVocabSize: int, numLayers: int) -> None:
        super(Transformer, self).__init__()
        self.encoderEmbedding = nn.Embedding(srcVocabSize, mDim)
        self.decoderEmbedding = nn.Embedding(tgtVocabSize, mDim)
        self.positionalEncoding = PositionalEmbedding(mDim, maxSeqLen)

        self.encoderLayers = nn.ModuleList([EncoderLayer(mDim, nHeads, ffDim, dropout) for _ in range(numLayers)])
        self.decoderLayers = nn.ModuleList([DecoderLayer(mDim, nHeads, ffDim, dropout) for _ in range(numLayers)])

        self.fc = nn.Linear(mDim, tgtVocabSize)
        self.dropout = nn.Dropout(dropout)

    def generateMask(self, src: Tensor, tgt: Tensor) -> tuple[Tensor, Tensor]:
        srcMask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgtMask = (src != 0).unsqueeze(1).unsqueeze(3)
        seqLength = tgt.size(1)
        noPeakMask = (1 - torch.triu(torch.ones(1, seqLength, seqLength), diagonal = 1)).bool()
        tgtMask = tgtMask & noPeakMask
        return srcMask, tgtMask

    def forward(self, src: Tensor, tgt: Tensor) -> tuple[Tensor, Tensor]:
        srcMask, tgtMask = self.generateMask(src, tgt)
        srcEmbedded = self.dropout(self.positionalEncoding(self.encoderEmbedding(src)))
        tgtEmbedded = self.dropout(self.positionalEncoding(self.encoderEmbedding(tgt)))

        encOutput = srcEmbedded
        for encLayer in self.encoderLayers:
            encOutput = encLayer(encOutput, srcMask)

        decOutput = tgtEmbedded
        for decLayer in self.decoderLayers:
            decOutput = decLayer(decOutput, encOutput, srcMask, tgtMask)

        output = self.fc(decOutput)
        return output

# Loading Datasets
csvPath = './time_series_15min_singleindex.csv'
df = pd.read_csv(csvPath)
featureList = ['utc_timestamp', 'DE_solar_generation_actual', 'DE_wind_onshore_profile',
               'DE_solar_generation_actual', 'DE_solar_profile', 'DE_wind_profile',
               'DE_wind_offshore_profile', 'DE_load_actual_entsoe_transparency',
               'DE_50hertz_wind_onshore_generation_actual', 
               'DE_transnetbw_wind_onshore_generation_actual'
               ]
targetList = ['DE_wind_generation_actual', 'DE_wind_onshore_generation_actual',
              'DE_wind_offshore_generation_actual']
features = df[featureList]
targets = df[targetList]
features = df[featureList].apply(pd.to_numeric, errors='coerce')
targets = df[targetList].apply(pd.to_numeric, errors='coerce')
features.fillna(0, inplace=True)  
targets.fillna(0, inplace=True)

# Data Split
trainFeatures, testFeatures, trainTargets, testTargets = train_test_split(features, targets, test_size = 0.2, random_state = 1)
testFeatures, validFeatures, testTargets, validTargets = train_test_split(testFeatures, testTargets, test_size = 0.5, random_state = 1)

# Dataset class for tabular data
class TabularDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features.values, dtype = torch.float32)
        self.targets = torch.tensor(targets.values, dtype = torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create datasets
trainSet = TabularDataset(trainFeatures, trainTargets)
valSet = TabularDataset(validFeatures, validTargets)

# DataLoader parameters
batchSize = 32

# Create DataLoaders
trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle = True)
valLoader = DataLoader(valSet, batch_size=batchSize, shuffle = True)

model = Transformer(mDim, nHeads, ffDim, 0.1, seqLength, 1000, 100, 10)
lossFunc = nn.MSELoss() 
criterion = optim.Adam(model.parameters(), lr = 0.0001)
scoreFuncs = {'Accuracy' : accuracy_score,
               'Recall' : lambda y_true, y_pred: recall_score(y_true, y_pred, average = 'macro'),
               'F1' : lambda y_true, y_pred: f1_score(y_true, y_pred, average = 'macro'),
               'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average = 'macro')}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
(trainResults, testResults, _) = runEpoch(model, trainLoader, valLoader, lossFunc,
                                          criterion, scoreFuncs, 100, device)
print(testResults)

