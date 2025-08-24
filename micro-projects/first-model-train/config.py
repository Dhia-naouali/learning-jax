from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()    
    
    config.data_dir = "data"
    config.download = True
    config.loader_prefetch_factor = 2
    config.train_batch_size = 32
    config.val_batch_size = 128
    config.epochs = 32
    config.init_learning_rate = 3e-4
    config.weight_decay = 1e-2
    
    return config