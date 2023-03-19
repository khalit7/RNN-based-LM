import constants as CONSTANTS
import torch
from random import choices


def generate_text(model,vocab,tokenizer,starting_text=""):
    text = starting_text
    tokens = vocab(tokenizer(starting_text))
    tokens = vocab(["<SOS>"]) + tokens
    
    tokens = torch.tensor(tokens)
    
    num_of_generated_tokens = 0
    while True:
        with torch.no_grad():
            dist = model(tokens.reshape(1,-1))[-1,:].softmax(dim=0)
            
        # select a word from dist
        generated_token = choices(range(len(dist)),dist,k=1)[0]

        tokens = torch.cat((tokens,torch.tensor([generated_token])))
        generated_word = vocab.lookup_token(generated_token)
        text += f" {generated_word}"
        num_of_generated_tokens += 1
        
        if generated_word == "<EOS>":
            break
            
        if num_of_generated_tokens == 100:
            break  
        
    return text