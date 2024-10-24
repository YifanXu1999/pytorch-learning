import torch
device = torch.device('mps')

print("----rand----")

a = torch.rand(2, 3)
b = torch.rand(2, 3)
print(a)
print(b)

print("----addition----")

print(a + b)

print(a.add(b))
print(torch.add(a,b))
print(a)
print(a.add_(b))
print(a)

print("----subtraction----")

print(a - b)

print(a.subtract(b))
print(torch.subtract(a,b))
print(a)
print(a.subtract_(b))
print(a)

print("----mul----")
a = torch.rand(2, 3)
b = torch.rand(2, 3)
print(a)
print(b)

print(a*b)
print(torch.multiply(a,b))
print(a.mul(b))


print("----div----")
a = torch.rand(2, 3)
b = torch.rand(2, 3)
print(a)
print(b)

print(a/b)
print(torch.div(a,b))
print(a.div(b))

print("----matrix multiplication----")

a = torch.ones(2, 1)
b = torch.ones(1,2)

print(a @ b)
print(a.matmul(b))
print(torch.matmul(a, b))

print("----higher order matrix multiplication----")

a = torch.ones(1, 2, 3, 4)
b = torch.ones(1, 2, 4, 3)

print(torch.matmul(a, b))

print("----pow---")
a = torch.tensor([1,2])
print(a)

print(torch.pow(a, 3))
print(a.pow(3))


print("----Exp---")
print(torch.exp(a))
print(a.exp())

print("----Log---")

print(torch.log(a))

print("----sqrt---")

print(torch.sqrt(a))

