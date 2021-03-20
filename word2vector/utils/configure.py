class Config():
    """This is config class that store all parameter about training model"""
    def __init__(self, vocab_size, bag_size):
        self.mode = True # True is skip gram mode
        self.vocab_size = vocab_size
        self.embedding_dim = 1024
        self.batch_size = 512
        self.bag_size = bag_size
        
        self.lr = 0.00001
        self.epochs = 30
        
        self.save_path = "./ckpt"