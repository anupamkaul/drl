import torch
import numpy as np

# create an uninitialized tensor of 3 x 2
a = torch.FloatTensor(3, 2)
print(a)

# no nvidia driver: cannot allocate GPU mem to tensor yet
#ca = a.cuda()
#print(ca)

import math
from tensorboardX import SummaryWriter

writer = SummaryWriter()
funcs = {"sin":math.sin, "cos":math.cos, "tan":math.tan} 

for angle in range(-360, 360):
	angle_rad = angle * math.pi / 180
	for name, fun in funcs.items():
		val = fun(angle_rad)
		writer.add_scalar(name, val, angle)
	writer.close()




