import torch
import torch.utils.data as data
import torch.nn as nn

from utils import configure
from model import word2vec

def train_step(model: word2vec.Word2VecModel, training_loader: data.DataLoader, device: torch.device, config: configure.Config) -> float:
    """Train model with steps 
    
    Args:
        model(word2vec.Word2VecModel): the model will be trained
        training_loader(torch.utils.data.Dataloader): the dataloader of training dataset
        device(torch.device): the hardware of training
        config(configure.Config): the training config
    
    Returns:
        float: avg loss in steps
    """
    model.train() # enter train mode to refresh gradient
    model.zero_grad()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    avg_loss = 0
    
    for step, (inputs, targets) in enumerate(training_loader):
        optimizer.zero_grad()
        
        # Skip-Gram:
        #     inputs: torch.Tensor(batch_size) to torch.Tensor(batch_size * bag_size)
        #     targets:list[torch.Tensor](bag_size, batch_size) to torch.Tensor(batch_size * bag_size)
        # CBOW:
        #     inputs: list[torch.Tensor](bag_size, batch_size) to torch.Tensor(batch_size, bag_size)
        #     targets: torch.Tensor(batch_size)
        
        if config.mode is True:
            bag_size = len(targets)
            inputs = torch.transpose(inputs.repeat(bag_size, 1), 0, 1).contiguous().view(-1)  
            targets = torch.transpose(torch.stack(targets), 0, 1).contiguous().view(-1)
        else:
            bag_size = len(inputs)
            inputs = torch.transpose(torch.stack(inputs), 0, 1).contiguous().view(-1)  
            #targets = torch.transpose(torch.repeat(bag_size, 1), 0, 1).contiguous().view(-1)
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        
        loss = loss_function(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        print("\r", "Train step[{}/{}] loss:{}]".format(step + 1, len(training_loader), avg_loss), end="")
    return avg_loss