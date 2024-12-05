import os
import argparse
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from data import *
from models import *
from trainer import *
from get_scores_and_prune import *


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Change for multi-node setups
    os.environ['MASTER_PORT'] = '12355'  # Ensure the port is free
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def preprocess_on_single_gpu(args):
    """
    Run preprocessing steps on a single GPU before distributed training.
    """
    print("Preprocessing on a single GPU...")

    # Parse arguments
    seed = args.seed
    sparsity = args.sparsity

    # Set random seed
    torch.manual_seed(seed)

    # Load configuration
    config = load_config("config.yaml")

    device = "cuda:0"  # Use GPU 0 for preprocessing
    sparse = config["sparse"]
    num_workers = config["num_workers"]
    dataset = config["dataset"]
    batch_size = config["batch_size"]
    checkpoint_path = config["checkpoint_path"]
    sparse_checkpoint_path = config["sparse_checkpoint_path"]

    # Set paths based on sparsity
    save_path = sparse_checkpoint_path if sparse else checkpoint_path

    # Prepare data
    train_loader, test_loader, train_samples = get_dataloader(dataset, batch_size, num_workers)

    if sparse:
        model = ResNet18().to(device)
        filepath = os.path.join(checkpoint_path, f"ckpt_{19}.pth")
        ckpt = torch.load(filepath, weights_only=True)
        model.load_state_dict(ckpt["net"])
        train_subset, train_samples = sparse_loader(
            train_loader, train_samples, model, device, sparsity, batch_size, num_workers
        )
        del model
    else:
        train_subset = train_loader
    
    _, test_data = load_data("cifar10")

    # Return preprocessing results
    return train_subset, test_data, train_samples, save_path, config


def test(rank, world_size, test_data, model):
    """
    Evaluate the model and compute accuracy.
    """
    test_loader = create_dataloader(rank, world_size, test_data, batch_size=128)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100. * correct / total
    if rank == 0:  # Log only from the main process
        print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def save_checkpoint(epoch, model, optimizer, accuracy, rank, save_path):
    """
    Save model state, optimizer state, and accuracy as a checkpoint.
    """
    if rank == 0:  # Save checkpoint only from the main process
        os.makedirs(save_path, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy
        }
        checkpoint_path = os.path.join(save_path, f'ckpt_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


def create_dataloader(rank, world_size, data, batch_size=128):
    sampler = DistributedSampler(data, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(data, batch_size=batch_size, sampler=sampler)
    return dataloader


def train(rank, world_size, train_subset, test_loader, config, save_path, num_epochs):
    """
    Distributed training function.
    """
    setup(rank, world_size)

    dataloader = create_dataloader(rank, world_size, train_subset, config["batch_size"])
    model = ResNet18().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(config["start_epoch"], config["start_epoch"] + num_epochs + 1):
        ddp_model.train()
        for batch_idx, (i, data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch} complete")

        # Test and save checkpoint
        accuracy = test(rank, world_size, test_loader, ddp_model)
        save_checkpoint(epoch, ddp_model, optimizer, accuracy, rank, save_path)

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Sparse Training Script")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for training")
    parser.add_argument("--sparsity", type=float, required=True, help="Sparsity level for sparse training")
    args = parser.parse_args()

    # Preprocessing on a single GPU
    train_subset, test_loader, train_samples, save_path, config = preprocess_on_single_gpu(args)

    # Start distributed training
    world_size = 4  # Number of GPUs
    num_epochs = config["num_epochs"]
    mp.spawn(train, args=(world_size, train_subset, test_loader, config, save_path, num_epochs), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
