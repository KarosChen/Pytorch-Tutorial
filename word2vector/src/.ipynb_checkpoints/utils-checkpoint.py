#!/usr/bin/env python
# coding: utf-8

# In[6]:


import csv
import string
import os
from . import modules


# In[7]:


def read_data(path_name: str) -> list:
    """read data from csv file
    
    There are three information in csv: title,abstract and classifications.
    This function extract abstract and split it to sentences without punctuation.
    
    Args:
        path_name: the path of data that will be loaded
    
    Returns:
        the sentences without punctuation
    """
    no_punc_sentences = []
    if os.path.isfile(path_name):
        with open(path_name, newline='') as train_csv:
            print("Reading data...")
            train_set = csv.DictReader(train_csv)
            abstract_set = [row["Abstract"] for row in train_set] # extract all abstracts to list
            for abstract in abstract_set:
                sentences = abstract.split(".") 
                for sen in sentences: 
                    for char in sen:
                        if char in string.punctuation:
                            sen = sen.replace(char, " ") # for each sentence, we use space to replace punctuation 
                    no_punc_sentences.append(sen)
    else:
        print("File is not exist!")
    return no_punc_sentences


# In[8]:


def save_model(model: modules.Word2VecModel, store_path: str, loss: float, epoch: int):
    """Save model
    
    Args:
        model: the model will be saved
        store_path: the store path
        loss: the model loss
        epoch: the model in which epoch
    """
    torch.save(model.state_dict(), "{}/model_{}_{:.3f}.ckpt".format(store_path, epoch, loss))


# In[ ]:


def load_model(model: modules.Word2VecModel, load_path: str) -> modules.Word2VecModel:
    """Load model
    
    Args: 
        model: the model will be saved
        load_path: the load path
    
    Returns:
        return model
    """
    print("Load model from {}".format(load_path))
    model.load_state_dict(torch.load("{}.ckpt".format(load_path)))
    return model


# In[ ]:


def find_nearest(word: int, top_nearest: int, model: modules.Word2VecModel) -> list:
    """Find nearest word
    
    Args:
        word: the index of word
        top_nearest: the number of nearest word
        model: model
    
    Returns:
        return list of indices of nearest words
    """
    parameters = model.embedding_layer.parameters()
    for para in parameters:
        para_data = para.data
    parameters = torch.Tensor(para_data)
    similarity_matrix = torch.mm(parameters, torch.transpose(parameters, 0, 1))
    similarity_vector = similarity_matrix[word]
    (values, indices) = torch.topk(similarity_vector, top_nearest)
    return indices


# In[ ]:




