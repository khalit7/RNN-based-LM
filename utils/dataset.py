import constants as CONSTANTS

import torch
from torch.utils.data import DataLoader
from torchtext.datasets import WikiText2
from torchtext.vocab import vocab
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer

from functools import partial

def _get_data_itr(split):
    data_itr = WikiText2(split=split)
    
    return data_itr


def _get_tokenizer():
    
    tokenizer = get_tokenizer("basic_english")
    
    return tokenizer

def _build_vocab(data_itr,tokenizer):
    v = build_vocab_from_iterator(map(tokenizer,data_itr),min_freq=CONSTANTS.min_freq,specials=["<unk>","<SOS>","<EOS>","<PAD>"])
    v.set_default_index(v["<unk>"])
    
    return v


def _collate_fn(batch,vocab,tokenizer):
    '''
    takes a batch of paragraphs from the dataset, returns a batch of X,Y
    '''
    X=[]
    Y=[]
    for b in batch:
        
        transformed_b = vocab(tokenizer(b))
        # if b is too short, skip it
        if len(transformed_b) < CONSTANTS.min_seq_len :
            continue
        
        # other wise chunk it up
        transformed_b_chunks = []
        for i in range(0,len(transformed_b),CONSTANTS.max_seq_len):
            chunk = transformed_b[i:i+CONSTANTS.max_seq_len]
            if len(chunk) > CONSTANTS.min_seq_len: 
                transformed_b_chunks.append( chunk )
        
        for chunk in transformed_b_chunks:
            # pad chunk and sos and eos
            if len(chunk) <= CONSTANTS.max_seq_len:
                num_of_pads = (CONSTANTS.max_seq_len-len(chunk))
                chunk =  vocab(["<SOS>"]) + chunk + vocab(["<EOS>"]) + vocab(["<PAD>"])*num_of_pads

                                            
            X.append(chunk[0:-1]) # all but the last
            Y.append(chunk[1:])   # all but the first
        
    X = torch.tensor(X,dtype=torch.long)
    Y = torch.tensor(Y,dtype=torch.long)
    
    return X,Y


def get_data_loader_and_vocab(data_split,batch_size=1,shuffle=True,vocab=None):
    
    # get data iterator:
    data_itr = _get_data_itr(data_split)
    
    # get tokenizer
    tokenizer = _get_tokenizer()
    
    # build vocab
    if vocab is None:
        vocab = _build_vocab(data_itr,tokenizer)
    
    dataloader = DataLoader(data_itr, batch_size=batch_size, shuffle=shuffle,collate_fn= partial(_collate_fn,vocab=vocab,tokenizer=tokenizer))
    
    return dataloader,vocab