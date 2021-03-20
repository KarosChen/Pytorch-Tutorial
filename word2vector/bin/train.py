def train(training_dataset, config):
    """The training process"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_loader = data.DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)
    
    model = modules.Word2VecModel(config.vocab_size, config.embedding_dim, config.batch_size, config.bag_size, config.mode)
    model = model.to(device)
    
    for epoch in range(config.epochs):
        print("Epoch [{}/{}]".format(epoch + 1, config.epochs))
        epoch_avg_loss = train_step(model, training_loader, device, config)
        save_model(model, config.save_path, epoch_avg_loss, epoch)