#%% Importing
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from ToTensor import ToTensor
from FilmDescriptionsDataset import FilmDescriptionsDataset
import time
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.utils import shuffle

#%% Defining Network

hidden_size = 150
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(64, hidden_size, 1, batch_first = True)
        self.linear = nn.Linear(hidden_size, 10)

    def forward(self, x, hidden, batch_size):
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, hidden_size)
        x = self.linear(lstm_out)
        x = F.softmax(x)
        
        # reshape to be batch_size first
        x = x.view(batch_size, -1) 
        x = x[:, -10:] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return x, hidden
    
    def init_hidden(self, batch_size):
        
        weight = next(self.parameters()).data
        
        hidden = (weight.new(1, batch_size, hidden_size).zero_(),
                  weight.new(1, batch_size, hidden_size).zero_())
        
        return hidden

#%% Testing Step
test_loss_vector = []
test_accuracy = []
def test(args, model, device, test_loader):
    model.eval()
    h = model.init_hidden(args.test_batch_size)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            data = sample['description']
            target = sample['genre']
            data, target = data.to(device), target.to(device)
            data = data.float()
            
            h = tuple([each.data for each in h])
            
            output, h = model(data, h, args.test_batch_size)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target.long()).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.long().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_loss_vector.append(test_loss)
    test_accuracy.append(100.*correct/len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(confusion_matrix(target.cpu(), pred.cpu()))

#%% Main
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch LSP')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=368, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)
    
    # https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    
    # test dataset
    data = pd.read_csv('dataset.cleaned.csv')
    shuffeld_data = shuffle(data).reset_index(drop=True)
    test_frac = 0.2
    test_data = shuffeld_data.iloc[int((1-test_frac)*len(data)):].reset_index(drop=True)
    
    test_data.to_csv('test_set_descriptions.csv')
    
    test_set = FilmDescriptionsDataset('test_set_descriptions.csv',
                                       transform=transforms.Compose([ToTensor()]))

    # test data loader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size,
                                              shuffle=True, drop_last = True, **kwargs)

    # make a network instance
    PATH = 'descript_retrain.pt'
    model = Net().to(device)
    model.load_state_dict(torch.load(PATH))

    test(args, model, device, test_loader)

#%% Calling main
if __name__ == '__main__':
    main()
