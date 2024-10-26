from cProfile import label
from multiprocessing.spawn import freeze_support
from sched import scheduler

import torch
import torch.nn as nn

import torchvision
from numpy import dtype
from torch.xpu import device

from vggnet import VGGNET

from load_data import train_data_loader, test_data_loader

device = torch.device("mps")


epoch_num = 200

lr = 0.01

net = VGGNET().to(device)

loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


for epoch in range(0, epoch_num):
    print("epoch is", epoch)
    net.train()

    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.int32)
        outputs = net(inputs)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print("step", i, " loss is:", loss.item())

        _,pred = torch.max(outputs, dim=1)

        correct = pred.eq(labels.data).cpu().sum().item()
        print("epoch is", epoch)
        print("lr is", optimizer.state_dict()["param_groups"][0]["lr"])

        print("step", i, "loss is:", loss.item(), "mini-batch correction is :", 100.0 * correct /10)

        scheduler.step()
