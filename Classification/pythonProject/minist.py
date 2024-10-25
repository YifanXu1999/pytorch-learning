from os import pread

import torch
import torchvision.datasets as dataset
import torchvision.transforms as transform

import torch
import sys

import torch.utils.data as data_utils
from sympy import false
from torch.xpu import device

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Print Python version
print(f"Python version: {sys.version}")

train_data = dataset.MNIST(root="mnist",
                           train=True,
                           download=True,
                           transform=transform.ToTensor())

test_data = dataset.MNIST(root="mnist",
                           train=False,
                           download=False,
                           transform=transform.ToTensor())

train_loader = data_utils.DataLoader(dataset=train_data, batch_size=64, shuffle=True)

test_loader = data_utils.DataLoader(dataset=test_data, batch_size=64, shuffle=True)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.fc = torch.nn.Linear(14 * 14  * 32, 10)


    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

device = torch.device('mps')

cnn = CNN().to(device)

loss_func = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("epoch is {}, ite is {}/{}, loss is {}".format(epoch+1, i, len(train_data) // 64, loss.item()))





    loss_test = 0
    accuracy  = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = cnn(images)
        loss_test += loss_func(outputs, labels)
        _, pred = outputs.max(1)
        accuracy += (pred == labels).sum().item()

    accuracy = accuracy / len(test_data)

    loss_test = loss_test

    print("epoch is {}, accuracy is {}, loss test is {}".format(epoch + 1, accuracy, loss_test.item()))


