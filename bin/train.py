import torch
import torch.utils.data as data

from utils import dataset
from utils import configure
from utils import executor
from utils import common
from model import word2vec

def train(training_dataset: dataset.Word2VecDataset, config: configure.Config):
    """The training process
    
    Args:
        training_dataset(utils.dataset.Word2VecDataset): the dataset will be trained
        config(utils.configure.Config): all parameter that would be used in training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_loader = data.DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)
    
    model = word2vec.init_word2vec_model(config.vocab_size, config.embedding_dim, config.batch_size, config.bag_size, config.mode)
    model = model.to(device)
    
    for epoch in range(config.epochs):
        print("Epoch [{}/{}]".format(epoch + 1, config.epochs))
        epoch_avg_loss = executor.train_step(model, training_loader, device, config)
        common.save_model(model, config.save_path, epoch_avg_loss, epoch)