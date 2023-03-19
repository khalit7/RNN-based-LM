import models
import torch
import torch.nn as nn         
        
    
    
def get_model_by_name(model_name,vocab_size,hidden_size):
    if "RNN-LM" in model_name:
        return models.RNN_LM(vocab_size,hidden_size)
    else:
        raise Exception("unrecognized model name")


def get_optimizer(model,lr):
    return torch.optim.Adam(model.parameters(),lr=lr)

def get_criterion():
    return nn.CrossEntropyLoss() 