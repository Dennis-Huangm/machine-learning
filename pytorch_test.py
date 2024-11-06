# Denis
# coding:UTF-8
import torch
from torch import nn
import functional as F
from tqdm import tqdm
import sys


def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, std=0.03)
        nn.init.normal_(layer.bias, std=0.01)
    elif type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, std=0.29)
        nn.init.normal_(layer.bias, std=0.01)


def evaluate_accuracy(data_iter, net):
    acc, time = 0, 0
    for x, y in data_iter:
        y_hat = net(x)
        cmp = y_hat.argmax(axis=1) == y
        acc += cmp.sum() / len(cmp)
        time += 1
    return acc / time


if __name__ == '__main__':
    batch_size = 256
    train_iter, val_iter = F.load_data_fashion_mnist(batch_size)
    num_features, epochs = 784, 50
    num_class = 10
    net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3), nn.ReLU(), nn.Conv2d(4, 8, 3),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2), nn.Conv2d(8, 32, 3), nn.ReLU(), nn.Conv2d(32, 64, 3), nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Flatten(), nn.Linear(4 * 4 * 64, num_class))
    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    trainer=torch.optim.SGD(net.parameters(),lr=0.001)

    for t in range(epochs):
        J,time=0,0
        for X, y in tqdm(train_iter,desc=f'epoch {t}/{epochs}',ascii=True, unit="batch",file=sys.stdout):
            y_hat = net(X)
            l = loss(y_hat, y)
            print(str(t)+''+str(l))
            time+=1
            J+=l
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
        acc = evaluate_accuracy(val_iter,net)
        #print(f'loss:{J/time}\n')
        print(f"=============== Test Accuracy:{acc} Loss:{J/time}===============")


