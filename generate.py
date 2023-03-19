import time
import random
import sys
sys.path.append("utils")
import text_generation
import helper
import dataset
import constants as CONSTANTS


if __name__ == "__main__":
    
    train_loader,vocab = dataset.get_data_loader_and_vocab("train",batch_size=1,shuffle=True,vocab=None)
    tokenizer = dataset._get_tokenizer()
    
    model = helper.get_model_by_name("RNN-LM",len(vocab),CONSTANTS.hidden_size)
    
    starting_text = input("write your starting text: ")
    print("generating ... ", end="",flush=True)
    generated_text = text_generation.generate_text(model,vocab,tokenizer,starting_text=starting_text)
    print("Done! ", flush = True)
    
    
    # print with delayed effect
    for word in generated_text.split():
        time.sleep(random.uniform(0.2, 0.7)) # sleep beetween 0.2 and 0.9 seconds
        print(word, end = " ",flush=True)