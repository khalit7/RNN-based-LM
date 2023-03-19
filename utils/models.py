import constants as CONSTANTS

import torch.nn as nn
import torch




class RNN_LM(nn.Module):
    
    def __init__(self,vocab_size,hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size,CONSTANTS.embed_size)
        
        self.rnn = nn.RNN(input_size=CONSTANTS.embed_size,hidden_size=self.hidden_size,num_layers=1,nonlinearity="tanh",batch_first=True)
        
        self.lin = nn.Linear(in_features=self.hidden_size, out_features=vocab_size)

    def forward(self,x):
        '''
        x has the shape (batch_size*context_size)
        output has the shape batch_size*vocab_size
        '''
        
        x = self.embed(x)                                           # shape is (batch_size*seq_len*CONSTANTS.embed_size)
        x,_ = self.rnn(x)                                             # shape is batch_size,seq_len,hidden_dim
        
        batch_size,seq_len,hidden_size = x.shape
        
        x = self.lin(x.reshape(-1,hidden_size))                # shape is batch_sizeXseq_len,vocab_size
        
        return x