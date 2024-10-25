import re
from turtle import Shape

import torch
import numpy as np
from sympy.codegen.fnodes import dimension

# -----------Load Data------------
data = []

ff = open("housing.data").readlines()
for item in ff:
    out = re.sub(r"\s{2,}", " ", item).strip()
    data.append(out.split(" "))

data = np.array(data).astype(np.float32)

print(data.shape)
# 0-n-1 columns are input features
X=data[:, 0:-1]
# Last column is the housing price
Y=data[:, -1]


# -----------Define training and testing data------------
X_train = X[0:496, ...]
Y_train = Y[0:496, ...]
X_test = X[496:, ...]
Y_test = Y[496:, ...]
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# --------- Define NN -------------

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 200)
        self.predict = torch.nn.Linear(200, n_output)

    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out

net = Net(13, 1)

loss_func = torch.nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for i in range(10000):
    x_data = torch.tensor(X_train, dtype=torch.float32)
    y_data = torch.tensor(Y_train, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data) * 0.0001

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print("ite: {}, loss: {}".format(i, loss* 10000))
    # print(pred[0:10])
    # print(y_data[0:10])

    x_data = torch.tensor(X_test, dtype=torch.float32)
    y_data = torch.tensor(Y_test, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    print(pred[0:10])
    print(y_data[0:10])
    loss_test = loss_func(pred, y_data) / 10
    print("ite: {}, loss_test:{}".format(i, loss_test))

torch.save(net, "model/model.pkl")
torch.save(net.state_dict(), "model/params.pkl")

