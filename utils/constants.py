import yaml
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)


max_seq_len = config["max_seq_len"]
min_seq_len = config["min_seq_len"]
min_freq = config["min_freq"]

model_name = config["model_name"]
batch_size = config["batch_size"]
epochs = config["epochs"]
learning_rate = config["learning_rate"]

embed_size = config["embed_size"]
hidden_size = config["hidden_size"]