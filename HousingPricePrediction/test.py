import re

import torch
import numpy as np
from torch.onnx.symbolic_opset9 import tensor


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

# -----------Load Data------------
data = []

ff = open("housing.data").readlines()
for item in ff:
    out = re.sub(r"\s{2,}", " ", item).strip()
    data.append(out.split(" "))

data = np.array(data).astype(np.float32)

# print(data.shape)
# 0-n-1 columns are input features
X=data[:, 0:-1]
# Last column is the housing price
Y=data[:, -1]

X_test = X[496:, ...]
Y_test = Y[496:, ...]

x_data = torch.tensor(X_test, dtype=torch.float32)
y_data = torch.tensor(Y_test, dtype=torch.float32)

model = Net(13,1)
model.load_state_dict(torch.load("model/params.pkl", weights_only=True))
eval = model.forward(x_data)

print(torch.squeeze(eval))
print(y_data)
