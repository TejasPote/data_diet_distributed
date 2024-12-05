import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


torch.utils.data.DistributedSampler:

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

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



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Change this to the master node's IP address if using multiple machines
    os.environ['MASTER_PORT'] = '12355'  # Pick a free port on the master node
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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

def save_checkpoint(epoch, model, optimizer, accuracy, rank, save_path=save_path):
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
        checkpoint_path = os.path.join(save_path, f'/ckpt_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


def create_dataloader(rank, world_size, data, batch_size=128):
    # transform = transforms.Compose([transforms.ToTensor()])
    # dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(data, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(data, batch_size=batch_size, sampler=sampler)
    return dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Sparse Training Script")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for training")
    parser.add_argument("--sparsity", type=float, required=True, help="Sparsity level for sparse training")
    return parser.parse_args()



def train(rank, world_size, epochs=5):

    setup(rank, world_size)
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

    train_loader, test_loader, train_samples = get_dataloader(dataset, batch_size, num_workers)

    if sparse:
        model = ResNet18()
        model = model.to(device)
        filepath = os.path.join(checkpoint_path, f"ckpt_{19}.pth")
        ckpt = torch.load(filepath, weights_only=True)
        model.load_state_dict(ckpt["net"])
        train_subset, train_samples = sparse_loader(train_loader,train_samples, model, device, sparsity, batch_size, num_workers)
        del model

    _ , test_data = load_data("cifar10")

    dataloader = create_dataloader(rank, world_size, train_subset, batch_size)
    model = ResNet18().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(start_epoch, start_epoch + num_epochs + 1):
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
        accuracy = test(rank, world_size, test_data, ddp_model)
        save_checkpoint(epoch, ddp_model, optimizer, accuracy, rank)
    
    cleanup()

def main():
    world_size = 4  # Number of GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()