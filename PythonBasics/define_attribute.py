import  torch
from torch.onnx.symbolic_opset9 import tensor

device = torch.device('mps')

a = torch.tensor([2, 2], device=device, dtype=torch.float32)

print(a)
print(a.dtype)

i = torch.tensor([[0, 1, 2], [0, 1, 2]])
v = torch.tensor([1, 2, 3])
a = torch.sparse_coo_tensor(i, v, (4, 4)).to_dense()
print(a)