class Config():
    """This is config class that store all parameter about training model"""
    def __init__(self, mode, vocab_size, bag_size):
        self.mode = mode # True is skip gram mode
        self.vocab_size = vocab_size
        self.embedding_dim = 250
        self.batch_size = 1024
        self.bag_size = bag_size
        
        self.lr = 0.00001
        self.epochs = 10
        
        self.save_path = "./ckpt"