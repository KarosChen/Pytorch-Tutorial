def test():
    """The test process"""
    window_size = 2
    training_set = read_data("./datasets/trainset.csv")
    
    training_dataset = Word2VecDataset(training_set, window_size, True)
    config = Config(len(training_dataset.word2idx), training_dataset.bag_size)
    model = Word2VecModel(config.vocab_size, config.embedding_dim, config.batch_size, config.bag_size, config.mode)
    model = load_model(model, "./module/model_29_0.095")

    #print(training_dataset.word2idx)
    word_idx = 14
    indices = find_nearest(word_idx, 8, model)
    nearest_words = [training_dataset.idx2word[i.item()] for i in indices]
    print(training_dataset.idx2word[word_idx])
    print(nearest_words)