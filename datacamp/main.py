import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
# import numpy as np

# sigmoid activation function to input tensor
tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
sig, act = nn.Sigmoid(), nn.Softmax(dim=-1)
print(f"Sigmoid applied: {sig(tensor)}")
print(f"Softmax applied: {act(tensor)}") # softmax expects all values in the tensor to be floating point(doesn't work with longs)

# one-hot encoding to convert single label to a tensor
one_hot_1 = F.one_hot(torch.tensor(0), num_classes=3)
one_hot_2 = F.one_hot(torch.tensor(1), num_classes=3)
one_hot_3 = F.one_hot(torch.tensor(2), num_classes=3)

print(one_hot_1)
print(one_hot_2)
print(one_hot_3)

# basic cross entropy loss calculation
predictions = torch.tensor([[0.1, 6.0, -2.0, 3.2]])
ground_truth = 2

# reason we pass [ground_truth] is to generate a tensor of size 1 x 4 and make it match the dimensions as the predictions tensor
encoded = F.one_hot(torch.tensor([ground_truth]), num_classes=4) # num_classes is same as predictions.shape[1]
func = CrossEntropyLoss()
# takes in predictions prior to the activation and the ground truth encoded tensor and outputs the loss value
print(f"loss: {func(predictions.float(), encoded.float())}")