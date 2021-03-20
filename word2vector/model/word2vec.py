import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

class Word2VecModel(nn.Module):
    """This is a word2vector model
    
    This model can produce embedding vector of the word
    that were used to train model
    
    Attributes:
        vocab_size(int): the total number of vocab
        embedding_dim(int): the dimension of embedding vecotor
        batch_size(int): the number of size were used to train or inference
        bag_size(int): the numbers of context vector
        skip_gram(bool): use skip_gram or CBOW mode
    """
    def __init__(self, vocab_size: int, embedding_dim: int, batch_size: int, bag_size: int, skip_gram: bool=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.bag_size = bag_size
        self.skip_gram = skip_gram
        
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.linear_layer = nn.Linear(self.embedding_dim, self.vocab_size)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Train the model
        
        Args: 
            inputs(torch.Tensor): training data (batch, bag_size)
        
        Returns:
            torch.Tensor: the prob distribution of vocab
        """
        if self.skip_gram:
            embedding_vectors = self.embedding_layer(inputs)       # (batch, embedding_dim)
        else:
            embedding_vectors = []
            inputs = inputs.view(-1)                               # (batch * bag_size)
            embedding_vectors = self.embedding_layer(inputs)       # (batch * bag_size, embedding_dim)
            embedding_vectors = embedding_vectors.view(self.batch_size, -1, self.embedding_dim) # (batch, bag_size, embedding_dim)
            embedding_vectors = torch.div(torch.sum(embedding_vectors, 1), self.bag_size) # (batch, embedding_dim)
        vocab_vectors = self.linear_layer(embedding_vectors)       #(batch, vocab_size)
        return vocab_vectors
    
    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """Inference the word to vector
        
        Args: 
            inputs(torch.Tensor): inferenced sequence (batch * seq_len): 
        
        Returns:
            torch.Tensor: embedding vectors of each input words
        """
        inputs = inputs.view(-1)
        embedding_vectors = self.embedding_layer(inputs) # (batch * seq_len, embedding_dim)
        embedding_vectors = embedding_vectors.view(self.batch_size, -1, self.embedding_dim) # (batch, seq_len, embedding_dim)
        return embedding_vectors

