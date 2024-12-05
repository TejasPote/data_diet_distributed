from get_scores_and_prune import *
from data import *
from models import *
from trainer import *
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing sparse data ...')
    train_loader, test_loader = get_dataloader('cifar10', 128, 2)
    sparsity = 0.5
    batch_size = 12
    num_workers = 2


    model = ResNet18() 
    cwd = os.getcwd()
    filepath = cwd + f"/checkpoint/ckpt_{19}.pth"
    ckpt = torch.load(filepath, weights_only = True)
    model.load_state_dict(ckpt['net'])
   

    sparse_train_loader, sparse_test_loader = sparse_loader(train_loader, model, device, sparsity, batch_size, num_workers)
   
    # Model
    print('==> Building model..')
    net = ResNet18()
    del ckpt
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.01,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    for epoch in range(start_epoch, start_epoch+20):
        train(epoch, net, optimizer,sparse_train_loader, device, criterion)
        test(epoch, net, sparse_test_loader, device, criterion)
        scheduler.step()

if __name__ == "__main__":
    main()