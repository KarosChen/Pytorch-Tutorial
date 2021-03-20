import torch
import torch.utils.data as data

from utils import dataset
from utils import configure
from utils import common
from model import word2vec

def test(loaded_model:str, testing_dataset:dataset.Word2VecDataset, word_idx:int, nearest_number: int):
    """test model output 
    
    In this function, we use find_nearest function to find nearest word 
    around specific word
    
    Args:
        loaded_model(str): the model path that would be loaded
        testing_dataset(dataset.Word2VecDataset): the testing dataset
        word_idx(int): the word will be tested
        nearest_number(int): the number of searched nearest word
    """
    #window_size = 2
    #training_set = read_data("./datasets/trainset.csv")
    #training_dataset = Word2VecDataset(data, window_size, skip_gram)
    
    config = configure.Config(len(testing_dataset.word2idx), testing_dataset.bag_size)
    model = init_word2vec_model(config.vocab_size, config.embedding_dim, config.batch_size, config.bag_size, config.mode)
    #model = load_model(model, "./module/model_29_0.095")
    model = common.load_model(model, loaded_model)
    
    #print(training_dataset.word2idx)
    indices = find_nearest(word_idx, nearest_number, model)
    nearest_words = [testing_dataset.idx2word[i.item()] for i in indices]
    #print(nearest_words.idx2word[word_idx])
    print(nearest_words)