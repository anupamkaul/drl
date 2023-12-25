import torch
import numpy as np

# 1. Creating tensors

# create an uninitialized tensor of 3 x 2
a = torch.FloatTensor(3, 2)
print(a)

# no nvidia driver: cannot allocate GPU mem to tensor yet
#ca = a.cuda()
#print(ca)

# clear the tensor content
a.zero_()
print(a)

# _ is inplace call. Operates on tensor content and returns
# same object/mem. Functional creates copy. 

# create using python iterable
b = torch.FloatTensor([1,2,3], [3,2,1]) 
print(b)

# create a zero object using numpy first
n = np.zeros(shape = (3, 2))
print(n)
b = torch.tensor(n)
print(b)


import math
from tensorboardX import SummaryWriter

writer = SummaryWriter()
funcs = {"sin": math.sin, "cos": math.cos, "tan": math.tan} 

for angle in range(-360, 360):
	angle_rad = angle * math.pi / 180
	for name, fun in funcs.items():
		val = fun(angle_rad)
		writer.add_scalar(name, val, angle)
writer.close()




