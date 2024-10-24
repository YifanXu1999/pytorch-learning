import torch

a = torch.Tensor([[1, 2], [3, 4]])

print(a)
print(a.type())
print(a.shape)

a = torch.ones(10, 10)
print(a)

a = torch.eye(10)
print(a)

a = torch.zeros(10,10)
print(a)

a = torch.zeros_like(a)
print(a)

a = torch.rand(2,2)
print(a)


a = torch.normal(mean=torch.zeros(5), std=torch.ones(5,5))
print(a)

a = torch.Tensor(2, 2).uniform_(-1, 1)
print(a)

a = torch.arange(0, 10, 0.1)

print(a)

a = torch.linspace(2, 10, 4)
print(a)

a = torch.randperm(10)
print(a)

