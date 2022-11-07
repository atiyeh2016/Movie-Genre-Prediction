#%% Importing
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import pathlib
import pandas as pd
import os
from FilmsDataset import FilmsDataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

#%%## Defining Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(48, 128, 5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(0.6)
        self.dropout2 = nn.Dropout2d(0.6)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(1024, 10)
        self.lrn = nn.LocalResponseNorm(2)
        self.pool1 = nn.MaxPool2d(5,2,2)
        self.pool2 = nn.MaxPool2d(3,2,1)        

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.lrn(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.lrn(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x

#%% Training Step
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        data = sample['image']
        target = sample['landmarks']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        data = data.float()
        output = model(data)
        loss = F.cross_entropy(output, target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#%%## Testing Step
predi = []
targeti = []
def test(args, model, device, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in valid_loader:
            data = sample['image']
            target = sample['landmarks']
            data, target = data.to(device), target.to(device)
#            data = data.float()
            output = model(data)
            valid_loss += F.cross_entropy(output, target.long(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            predi.append(pred)
            targeti.append(target)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
    valid_loss /= len(valid_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / (len(valid_loader.dataset))))

#%% Main
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch LSP')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)
    
    # https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
    kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}

    # train dataset
    parent_path = pathlib.Path().absolute().parent
    folder1 = 'train_images'
    folder2 = 'train_set'
    img_path = os.path.join(parent_path, folder1, folder2)
    data = pd.read_csv('dataset.csv')
    shuffeld_data = shuffle(data).reset_index(drop=True)
    test_frac = 0.2
    train_data = shuffeld_data.iloc[:int((1-test_frac)*len(data))]
    test_data = shuffeld_data.iloc[int((1-test_frac)*len(data)):].reset_index(drop=True)
    
    train_data.to_csv('train_set.csv')
    test_data.to_csv('test_set.csv')
    
    transforms_list = transforms.Compose([
                                         transforms.Resize((200,200)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0,0,0), (1,1,1))
                                         ])
    
    train_dataset = FilmsDataset('train_set.csv', img_path, transform=transforms_list)
    test_dataset = FilmsDataset('test_set.csv', img_path, transform=transforms_list)
    
    # train data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, **kwargs)

    # test data loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              shuffle=True, **kwargs)

    # make a network instance
    model = Net().to(device)

    # configure optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr,  weight_decay=1e-5)
#    torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # configure learning rate scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)          
            
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        
#        scheduler.step()

    # save the trained model
    if args.save_model:
	    torch.save(model.state_dict(), "films.pt")

    # run inference
#    if args.save_model:
#	    model = Net()
#	    model.load_state_dict(torch.load("lsp_cnn.pt"))
#	    sample = next(iter(valid_loader))
#	    model.eval()
#	    outputs = model(sample['image'].float())
#	    _, lbl = torch.max(outputs.data, 1)
#	    print('\nThe true lable: ', sample['landmarks'][0].item())
#	    print('The classifier lable: ', lbl[0].item())

#%% Calling main
if __name__ == '__main__':
    main()
