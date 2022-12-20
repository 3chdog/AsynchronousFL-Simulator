import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary
from tqdm import tqdm
import numpy as np
import os
import time
import argparse

# self modules
import models


class clientInstance():
    def __init__(self, batchSize=1024, lr=1e-3, epochs=1, device=None, datasetName=None):
        self.setBatchSize(batchSize)
        self.setLearningRate(lr)
        self.setEpochs(epochs)
        self.setDevice(device)
        # set once or we can change these ??
        self.setDatasetName(datasetName)
        self.setDataLoader()
        self.setModel()
        self.setLossAndOpt(self.Net)
    
    def setBatchSize(self, batchSize):
        self.batchSize = batchSize
    
    def setLearningRate(self, lr):
        self.lr = lr
    
    def setEpochs(self, epochs):
        self.epochs = epochs
    
    def setDevice(self, device):
        if device==None: self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else: self.device =device

    def setDatasetName(self, datasetName):
        self.DatasetName = datasetName

    def setDataLoader(self, datasetName=None):
        transform = transforms.ToTensor()
        if datasetName == None: self.DatasetName = "lenet"
        if datasetName == None: train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = self.batchSize, shuffle=32, num_workers = 2)
        if datasetName == None: test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = self.batchSize, shuffle=32, num_workers = 2)
        self.train_dataset, self.test_dataset, self.train_loader, self.test_loader = train_dataset, test_dataset, train_loader, test_loader

    def setModel(self, model=models.LeNet()):
        Net = model
        Net = Net.to(self.device)
        summary(Net, self.train_dataset[0][0].shape)
        self.Net = Net
    
    def setLossAndOpt(self, net):
        # Loss and Optimizer
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay = 1e-8)
        self.loss_function, self.optimizer = loss_func, optimizer
    
    def setParameters(self, stateDict):
        self.Net.load_state_dict(stateDict)
    
    def getParameters(self):
        return self.Net.state_dict()
    
    # train and return parameters
    def train(self, epochs=1, delaytime=0):
        self.train_elastic(self.Net, self.train_loader, self.optimizer, self.loss_function, epochs)
        time.sleep(delaytime)
        return self.Net.state_dict()

    def train_elastic(self, net, train_loader, optimizer, loss_function, epochs=1):
        print("Client Go Training.")
        net.train()
        for epoch in range(epochs):
            with tqdm(train_loader, unit="batch") as loader_t:
                for batch_idx, (image, label) in enumerate(loader_t):
                    image = image.to(self.device)
                    label = nn.functional.one_hot(label, 10)
                    label = label.to(torch.float)
                    label = label.to(self.device)

                    optimizer.zero_grad()
                    output = net(image)
                    loss = loss_function(output, label)
                    loss.backward()
                    optimizer.step()
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch+1,
                            batch_idx * len(image),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            loss.item()))

                    if (epoch+1)%20==0 or epoch==epochs-1:
                        torch.save(net.state_dict(), "./{}_{}_{}.pt".format(os.getpid(), self.DatasetName, epoch+1))  # <------ Store Models
    
    def eval(self):
        self.eval_elastic(self.Net, self.test_loader, self.loss_function)
    
    # not loaded into self.Net
    def eval_fromStateDict(self, stateDict=None):
        if stateDict==None: model = self.Net
        else:
            model = models.LeNet()
            model.to(self.device)
            model.load_state_dict(stateDict)

        self.eval_elastic(model, self.test_loader, self.loss_function)

    def eval_elastic(self, net, test_loader, loss_function):
        print("Client Go Evaluating.")
        net.eval()
        test_loss = 0
        correct = 0

        for (data, label) in test_loader:
            label_onehot = nn.functional.one_hot(label, 10)
            label_onehot = label_onehot.to(torch.float)
            label_onehot = label_onehot.to(self.device)
            data, label = data.to(self.device), label.to(self.device)

            with torch.no_grad():
                output = net(data)
                test_loss += loss_function(output, label_onehot).item()
                pred = output.data.max(1)[1] # get the index of the max log-probability
                correct += pred.eq(label.data).cpu().sum()


        test_loss = test_loss
        test_loss /= len(test_loader) # loss function already averages over batch size
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def getArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False, default="localhost", help='Input the server ip. e.g. 127.0.0.1')
    parser.add_argument('-p', '--port', type=int, required=False, default=8080, help='Input the server port. e.g. 8080')
    parser.add_argument('-d', '--delay', type=int, required=False, default=0, help='Input the delay time (second) of this client. e.g. 5')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = getArguments()
    server_url = "http://{}:{}".format(args.url, args.port)
    print("Server at {}".format(server_url))
    # Hyper Parameters
    np.random.seed(41)
    batch_size = 1024
    lr = 1e-3
    epochs = 3
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    client1 = clientInstance(batch_size, lr, epochs, device)
    mymodel = client1.train(client1.epochs, args.delay)

    # evaluate current client's model
    client1.eval()
    # load parameters into model and evaluate (but not change self.Net)
    client1.eval_fromStateDict(mymodel)
    # get Parameters (that can upload for server)
    print(client1.getParameters())
    print(client1.getParameters()['conv1.weight'])