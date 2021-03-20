#!/usr/bin/env python
# coding: utf-8

# In[30]:


import torch
import torch.utils.data as data


# In[77]:


class Word2VecDataset(data.Dataset):
    """Make pair word dataset

    Attributes:
        data: the data are maked to dataset
        window_size: the size of slipped distance
        bag_size: the size of context number
        skip_gram: skip_gram or CBOW mode
        pair_word_data: the pair of inputs and targets
        word2idx: word map to index dict
        idx2word: index map to word dict
        word_prob: every word probability
    """
    def __init__(self, data: list, window_size: int, skip_gram: bool=True):
        self.data = data
        self.window_size = window_size
        self.bag_size = window_size * 2 + 1
        self.skip_gram = skip_gram
        self.pair_word_data = []
        self.word2idx = {}
        self.idx2word = {}
        self.word_prob = []
        self.make_dict()
        self.calculate_sampling_prob()
        self.generate_pair_word()
        
    def generate_pair_word(self):
        """Generate pair of inputs and targets"""
        print("Generating pair word data...")
        for sentence in self.data:
            words = sentence.split(" ")
            word_len = len(words)
            for idx in range(word_len): # get left and right side output word of skip word
                lower_idx = (idx - self.window_size) if idx >= self.window_size else 0
                upper_idx = (idx + self.window_size) if idx <= (word_len - 1) - self.window_size else (word_len - 1)
                BOW = []
                if idx - self.window_size < 0:
                    pad_num = self.window_size - idx
                    for i in range(pad_num):
                        BOW.append(self.word2idx["<PAD>"])
                for offset in range((upper_idx - lower_idx) + 1):
                    nearest_idx = lower_idx + offset
                    if nearest_idx != idx:
                        BOW.append(self.word2idx[words[nearest_idx]])
                if idx + self.window_size > word_len - 1:
                    pad_num = self.window_size + idx - (word_len - 1)
                    for i in range(pad_num):
                        BOW.append(self.word2idx["<PAD>"])
                if self.skip_gram: # use skip gram or CBOW mode
                    self.pair_word_data.append((self.word2idx[words[idx]], BOW))
                else:
                    self.pair_word_data.append((BOW, self.word2idx[words[idx]]))
    
    def make_dict(self):
        """Make word2idx and idx2word dict"""
        print("Makeing dictionary...")
        self.word2idx["<PAD>"] = len(self.word2idx)
        self.idx2word[len(self.word2idx)] = "<PAD>"
        self.word_prob.append(0)
        for sentence in self.data:
            words = sentence.split(" ")
            for word in words:
                if word not in self.word2idx:
                    word_idx = len(self.word2idx)
                    self.word2idx[word] = word_idx
                    self.idx2word[word_idx] = word
                    self.word_prob.append(0)
                self.word_prob[int(self.word2idx[word])] += 1
                
    def calculate_sampling_prob(self):
        """Calculate the probability of word"""
        self.word_prob = [prob * 0.75 for prob in self.word_prob]
        prob_sum = 0
        for prob in self.word_prob:
            prob_sum += prob
        self.word_prob = [prob / prob_sum for prob in self.word_prob]
        
    def __getitem__(self, idx: int) -> list:
        """Get the item in dataset"""
        return self.pair_word_data[idx]
    
    def __len__(self) -> int:
        """Get the number of items in dataset"""
        return len(self.pair_word_data)

