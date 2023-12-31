# https://pytorch.org/tutorials/beginner/saving_loading_models.html

import torch.nn as nn
import numpy as np

# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
from torch import optim
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


# saving and loading the model for inference:

# save, using state_dict:
PATH = "./models/mymodel"

import torch
torch.save(model.state_dict(), PATH)
print("Model saved to ", PATH)

# reload:

#modelnew = TheModelClass(*args, **kwargs) # general

print("Model Loaded from ", PATH)
modelnew = TheModelClass()
modelnew.load_state_dict(torch.load(PATH))
modelnew.eval()

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in modelnew.state_dict():
    print(param_tensor, "\t", modelnew.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

print("\nfull model details: ", modelnew) 

# ----------------------------------

# save full model
PATH_FULLMODEL = "./models/mymodelfull"

torch.save(model, PATH_FULLMODEL)
print("Full Model saved to ", PATH_FULLMODEL)

print("Full Model Loaded from ", PATH_FULLMODEL, "\n")
modelfull = TheModelClass()
modelfull = torch.load(PATH_FULLMODEL)
modelfull.eval()
print(modelfull)

