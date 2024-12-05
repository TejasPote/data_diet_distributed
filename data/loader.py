import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn


transform = transforms.Compose(
          [transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )

class MyDataset(Dataset):
    def __init__(self, data):
        
        self.data = data 


    def __getitem__(self, idx):
        image = self.data[idx][0]

        label = self.data[idx][1]
        return idx, image, label
    def __len__(self):
        return len(self.data)

def load_data(dataset):
    if dataset == 'cifar10':
        train = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
        train = MyDataset(train)
        test = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)

    return train, test
        
def get_dataloader(dataset, batch_size, num_workers):
    if dataset == 'cifar10':
        train, test = load_data(dataset)    
        train_samples = len(train)
    
    train_loader  = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_samples