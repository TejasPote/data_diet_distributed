import torch
import os


def train(epoch, net, optimizer,trainloader, device, criterion):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (idx, inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        # preds = torch.nn.functional.softmax(outputs, dim = 1)
        
        # e = preds - torch.nn.functional.one_hot(targets, num_classes = 10)
        # el2n = e.norm(dim = 1, p = 2)
        # for i,s in list(zip(idx, el2n)):
            
        #     EL2N.update({i.item() : s.item()})

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    
    

def test(epoch, net, testloader, device, criterion, save_path):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
        acc = 100.*correct/total
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path+ f'/ckpt_{epoch}.pth')
    
    

    


