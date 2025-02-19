
import torch
print("torch version:",torch.__version__)
x = torch.rand(5, 3)
print(x)
# gpu
print("gpu:",torch.cuda.is_available())
# 查看CUDA版本
print('CUDA version:', torch.version.cuda)