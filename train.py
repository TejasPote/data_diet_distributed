import os
import argparse
import yaml
from data import *
from models import *
from trainer import *
from get_scores_and_prune import *
import torch
import torch.nn as nn
import torch.optim as optim

# Function to load configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Argument parser for seed and sparsity
def parse_args():
    parser = argparse.ArgumentParser(description="Sparse Training Script")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for training")
    parser.add_argument("--sparsity", type=float, required=True, help="Sparsity level for sparse training")
    return parser.parse_args()

# Main function
def main():
    # Parse arguments
    args = parse_args()
    seed = args.seed
    sparsity = args.sparsity

    # Set random seed
    torch.manual_seed(seed)

    # Load configuration
    config = load_config("config.yaml")

    device = config["device"]
    sparse = config["sparse"]
    num_workers = config["num_workers"]
    dataset = config["dataset"]
    batch_size = config["batch_size"]
    start_epoch = config["start_epoch"]
    num_epochs = config["num_epochs"]
    lr = config["lr"]
    momentum = config["momentum"]
    weight_decay = config["weight_decay"]
    checkpoint_path = config["checkpoint_path"]
    sparse_checkpoint_path = config["sparse_checkpoint_path"]

    # Set paths based on sparsity
    save_path = sparse_checkpoint_path if sparse else checkpoint_path

    # Prepare data
    print("==> Preparing data..")
    train_loader, test_loader, train_samples = get_dataloader(dataset, batch_size, num_workers)

    if sparse:
        model = ResNet18()
        model = model.to(device)
        filepath = os.path.join(checkpoint_path, f"ckpt_{19}.pth")
        ckpt = torch.load(filepath, weights_only=True)
        model.load_state_dict(ckpt["net"])
        train_loader, train_samples = sparse_loader(train_loader,train_samples, model, device, sparsity, batch_size, num_workers)
        del model

    

    # Build model
    print("==> Building model..")
    net = ResNet18()
    net = net.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs + 1):
        train(epoch, net, optimizer, train_loader, device, criterion)
        test(epoch, net, test_loader, device, criterion, save_path)
        scheduler.step()

if __name__ == "__main__":
    main()
