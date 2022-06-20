from agent import train
from model import LinearQNet
import torch.optim as optim
import torch.nn as nn
from pygame_env import room_types

learning_rates = [0.1, 0.01, 0.001]
optimizers = ['Adam', 'Adagrad', 'SGD']

criterion = nn.MSELoss()
model = LinearQNet(13, 512, 256, 8)


plot_count = 1
simulationnr_stop = 2

for room in room_types[0:]:
    for lr in learning_rates:
        for op in optimizers:
            if op == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif op == 'Adagrad':
                optimizer = optim.Adagrad(model.parameters(), lr=lr)
            else: 
                optimizer = optim.SGD(model.parameters(), lr=lr)
            
            print("learning rate is ", lr, "optimizer is", op , "room type is ", room)
            train(model, lr, optimizer, criterion, room, plot_count, simulationnr_stop) 
            plot_count += 1    
            