# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 12:03:20 2019

@author: Saranya
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import product

torch.set_printoptions(linewidth = 120)
torch.set_grad_enabled(True)



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)
        
        self.fc1 = nn.Linear(in_features= 12 * 4*4 , out_features = 120)
        self.fc2 = nn.Linear(in_features = 120 , out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)
    
    def forward(self,t):
        # (1) input layer
        t = t
        
        # (2) hidden convolution layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # (3) hidden convolution layer 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        
        # (4) hidden linear layer 1
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        
        # (5) hidden linear layer 2
        t = self.fc2(t)
        t = F.relu(t)
        
        # (6) output layer
        t = self.out(t)
        
        return t
    
train_set = torchvision.datasets.FashionMNIST(
            root = './data/FashionMNIST',
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor()
            ]))
    

    
train_loader =  torch.utils.data.DataLoader(train_set, batch_size =100, shuffle = True)

#Training loop

parameters = dict(
        lr = [0.01,0.001],
        batch_size = [1000,10000],
        shuffle = [True, False]
        )

for lr, batch_size, shuffle in parameters(*param_values):
    comment = f'batch_size ={batch_size} lr ={lr} shuffle = {shuffle}'

    network = Network()
    train_loader =  torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    optimizer = optim.Adam(network.parameters(), lr = lr)
    
    images,labels =  next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)
    
    tb = SummaryWriter(comment = comment)
    tb.add_image('images',grid)
    tb.add_graph(network, images)
    
    
    for epoch in range(10):
        total_loss = 0
        total_correct =0
        
        for batch in train_loader:
            images,labels = batch
            
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss +=loss.item() * batch_size
            total_correct +=get_num_correct(preds, labels)
        
        tb.add_scalar('Loss:', total_loss, epoch)
        tb.add_scalar('Number Correct:', total_correct, epoch)
        tb.add_scalar('Accuracy:', total_correct/len(train_set), epoch)
        
    
    
        for name, weight in network.named_parameters():
            tb.add_histogram(name, weight, epoch)
            
        
        print('epochs:', epoch ,'total_loss:', total_loss,'total_correct:', total_correct)
    
    tb.close()
                
    
## Using runmanager to train the model

class RunManager():
    def __init__(self):
        self.epoch_count =0
        self.epoch_loss =0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        
        self.run_params = None
        self.run_count =0
        self.run_data =[]
        self.run_start_time = None
        
        self.network =None
        self.loader = None
        self.tb = None
        
    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()
        
        self.run_params = run
        self.run_count +=1
        
        self.network = Network
        self.loader =  loader
        self.tb = SummaryWriter(comment = f'-{run}')
        
        images, labels =next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images' ,grid)
        self.tb.add_graph(self.network, images)
        
    def end_run(self):
        self.tb.close()
        self.epoch_close = 0
        
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count +=1
        self.epoch_loss = 0
        self.epoch_num_correct =0
        
    def end_epoch(self):
        epoch_duration =  time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy= self.epoch_num_correct / len(self.loader.dataset)
        
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad',param.grad, self.epoch_count)
    
    
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration
        
        for k, v in self.run_params.__asdict().items():
            results[k] = v
        
        self.run_data.append(results)
        
        df = pd.DataFrame.from_dict(self.run_data, orient ='columns')
        
        #below statements are specific to jupypter notebook
        clear_output(wait = True)
        display(df)
        
    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size
    
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct +=self._get_num_correct(preds, labels)
        
    @torch.no_grad()
    def _get_num_correct(preds, labels):
        return preds.argmax(dim =1).eq(labels).sum().item()
    
    def save(self, fileName):
        pd.DataFrame.from_dict(
                self.run_data,
                orient ='columns'
                ).to_csv(f'{fileName}.csv')
        
        with open(f'{fileName}.json','w',encoding ='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii = False, indent = 4)
        
        
params = OrderedDict(
        lr=[0.01],
        batch_size = [1000,10000])

m = RunManager()
for run in RunBuilder.get_runs(params):
    network = Network()
    loader = DataLoader(train_set, batch_size= run.batch_size)
    optimizer = optim.Adam(network.parameters(), lr = run.lr)
    
    m.begin_run(run, network, loader)
    for epoch in range(5):
        m.begin_epoch()
        for batch in loader:
            images, labels = batch
            preds =network(images)
            loss =F.cross_entropy(preds, labels)
            optimizer = zero.grad()
            loss.backward()
            optimizer.step()
            
            m.track_loss(loss)
            m.track_num_correct(preds, labels)
            
        m.end_epoch()
    m.end_run()
m.save('results')


