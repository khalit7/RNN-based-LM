import sys
sys.path.append("utils")
import dataset
import models
import constants as CONSTANTS
import helper
from trainer import Trainer
import torch

from torch.utils.tensorboard import SummaryWriter




if __name__ == "__main__":
    model_name = CONSTANTS.model_name
    batch_size = CONSTANTS.batch_size
    train_loader,vocab = dataset.get_data_loader_and_vocab("train",batch_size=batch_size,shuffle=True,vocab=None)
    val_loader,_ = dataset.get_data_loader_and_vocab("test",batch_size=batch_size,shuffle=True,vocab=vocab)
    model = helper.get_model_by_name(model_name,vocab_size = len(vocab),hidden_size=CONSTANTS.hidden_size)


    number_of_epochs = CONSTANTS.epochs
    criterion = helper.get_criterion()
    scheduler = None
    lr = CONSTANTS.learning_rate
    optimizer = helper.get_optimizer(model,lr)
    device = torch.device("mps" if torch.has_mps else "cpu")
    model_path = f"weights/{model_name}"

    writer = SummaryWriter(f".runs/{model_name}")

    trainer = Trainer(model,train_loader,val_loader,number_of_epochs,criterion,optimizer,scheduler,device,model_path,model_name,writer)
    
    trainer.train()