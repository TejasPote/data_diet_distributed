import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
from data import *

def sparse_loader(train_loader, train_samples, net, device, sparsity, batch_size, num_workers):

    scores = []
    for batch_idx, (idx, input, target) in enumerate(train_loader):

        print(f"Computing scores for batch idx {batch_idx} ....")
        input, target = input.to(device), target.to(device)
        output = net(input)
        preds = torch.nn.functional.softmax(output, dim = 1)
        e = preds - torch.nn.functional.one_hot(target, num_classes = 10)
        el2n = e.norm(dim = 1, p = 2)
        for i, s in zip(idx, el2n):
            scores.append((i.item(), s.item()))
        
    samples = int((1-sparsity)*train_samples)
    indices = sorted(scores, key = lambda x : x[1],  reverse = True)[:samples]
    indices = [k for k,v in indices]

    train_dense, _ = load_data('cifar10')
    train_subset = torch.utils.data.Subset(train_dense, indices)

    assert (len(train_subset) == samples)
    print(len(train_subset))

    sparse_train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return sparse_train_loader, samples


