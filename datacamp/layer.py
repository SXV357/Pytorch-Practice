# import torch
import torch.nn as nn

layer = nn.Linear(8, 5)

print(f"Weights:")
print(layer.weight)

print("After initializing weights using uniform distribution")
nn.init.uniform_(layer.weight)

print(layer.weight)