import os
import argparse
import yaml
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import psutil
import GPUtil
import matplotlib.pyplot as plt
from data import *
from models import *
from trainer import *
from get_scores_and_prune import *


def log_resource_utilization(log_file, interval=1):
    """
    Logs CPU and GPU utilization periodically in a clean, parseable format.
    """
    cpu_utilization = []
    gpu_utilization = []
    gpu_memory_utilization = []

    try:
        while True:
            # Log CPU utilization
            cpu_percent = psutil.cpu_percent()
            cpu_utilization.append(cpu_percent)

            # Log GPU utilization using torch
            if torch.cuda.is_available():
                gpu_percent = [
                    torch.cuda.utilization(gpu_id) for gpu_id in range(torch.cuda.device_count())
                ]
                gpu_memory = [
                    torch.cuda.memory_allocated(gpu_id)
                    / torch.cuda.get_device_properties(gpu_id).total_memory
                    * 100
                    for gpu_id in range(torch.cuda.device_count())
                ]
            else:
                gpu_percent = []
                gpu_memory = []

            gpu_utilization.append(gpu_percent)
            gpu_memory_utilization.append(gpu_memory)

            # Write to log file with clear formatting
            with open(log_file, "a") as f:
                f.write(f"CPU: {cpu_percent:.2f}%\n")
                for i, (util, mem) in enumerate(zip(gpu_percent, gpu_memory)):
                    f.write(f"GPU{i} Utilization: {util:.2f}%, GPU{i} Memory: {mem:.2f}%\n")
            time.sleep(interval)
    except KeyboardInterrupt:
        return cpu_utilization, gpu_utilization, gpu_memory_utilization




def save_runtime(start_time, end_time, log_file="runtime_log.txt"):
    runtime = end_time - start_time
    with open(log_file, "w") as f:
        f.write(f"Total runtime: {runtime:.2f} seconds\n")


def plot_utilization(cpu_utilization, gpu_utilization, gpu_memory_utilization):
    """
    Creates individual plots for CPU and each GPU utilization over time.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)  # Ensure a directory for saving plots

    # Plot CPU utilization
    plt.figure(figsize=(8, 4))
    plt.plot(cpu_utilization, label="CPU Utilization (%)", color="blue")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Utilization (%)")
    plt.title("CPU Utilization Over Time")
    plt.savefig("plots/cpu_utilization_plot.png")
    plt.show()

    # Plot GPU utilization
    for i, gpu_util in enumerate(gpu_utilization):
        plt.figure(figsize=(8, 4))
        plt.plot(gpu_util, label=f"GPU {i} Utilization (%)", color="green")
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Utilization (%)")
        plt.title(f"GPU {i} Utilization Over Time")
        plt.savefig(f"plots/gpu_{i}_utilization_plot.png")
        plt.show()


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

    # Measure total runtime
    start_time = time.time()

    # Log resource utilization in a separate process
    utilization_log_file = "utilization_log.txt"
    utilization_process = mp.Process(
        target=log_resource_utilization, args=(utilization_log_file,)
    )
    utilization_process.start()

    # Preprocessing on a single GPU
    train_subset, test_loader, train_samples, save_path, config = preprocess_on_single_gpu(args)

    # Start distributed training
    world_size = 4  # Number of GPUs
    num_epochs = config["num_epochs"]
    mp.spawn(train, args=(world_size, train_subset, test_loader, config, save_path, num_epochs), nprocs=world_size, join=True)

    # Stop utilization logging
    utilization_process.terminate()

    end_time = time.time()
    save_runtime(start_time, end_time)

    # Load utilization logs
    with open(utilization_log_file, "r") as f:
        logs = f.readlines()

    # Parse utilization logs
    cpu_utilization = []
    gpu_utilization = [[] for _ in GPUtil.getGPUs()]
    gpu_memory_utilization = [[] for _ in GPUtil.getGPUs()]

    

    for log in logs:
        log = log.strip()

        # Parse CPU utilization
        if log.startswith("CPU:"):
            try:
                cpu_percent = float(log.split(": ")[1][:-1])
                cpu_utilization.append(cpu_percent)
            except ValueError:
                print(f"Error parsing CPU utilization in log: {log}")
        
        # Parse GPU utilization
        elif log.startswith("GPU"):
            try:
                parts = log.split(", ")
                gpu_id = int(parts[0].split(" ")[0][3:])  # Extract GPU index
                gpu_util = float(parts[0].split(": ")[1][:-1])  # Extract GPU utilization
                gpu_mem = float(parts[1].split(": ")[1][:-1])  # Extract GPU memory usage
                
                gpu_utilization[gpu_id].append(gpu_util)
            except (IndexError, ValueError):
                print(f"Error parsing GPU utilization in log: {log}")
                
            gpu_utilization[gpu_id].append(gpu_util)
            gpu_memory_utilization[gpu_id].append(gpu_mem)

    # Plot utilization
    plot_utilization(cpu_utilization, gpu_utilization, gpu_memory_utilization)


if __name__ == "__main__":
    main()
