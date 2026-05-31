import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

input_tensor = torch.tensor([[3.75, 4.06, 1.23, -0.36]])

model = nn.Sequential(
    nn.Linear(4, 10), # 10(4) + 10 = 50 params
    nn.Linear(10, 5), # 5(10) + 5 = 55 params
    nn.Linear(5, 4) # 4(5) + 4 = 24 params
)

# total of 50 + 55 + 24 params = 129 parameters

def calc_params(model):
    res = 0
    for param in model.parameters():
        # params in 1 layer: weights + bias
            # number of elements in each tensor in the 2D weights tensor
            # number of elements in the bias tensor
        res += param.numel()

    print(f"total number of params: {res}")

# layers can be indexed as such like this
print(f"layer 1 params: {model[0].weight, model[0].bias}")
print(f"layer 2 params: {model[1].weight, model[1].bias}")
print(f"layer 3 params: {model[2].weight, model[2].bias}")

# each layer also has a .grad attribute which can be accessed(these are the values updated based on the gradients calculated off the loss function
print(f"Before backpropagation: ")
print(f"layer 1 gradients: {model[0].weight.grad, model[0].bias.grad}")
print(f"layer 2 gradients: {model[1].weight.grad, model[1].bias.grad}")
print(f"layer 3 gradients: {model[2].weight.grad, model[2].bias.grad}")

predictions = model(input_tensor)
ground_truth = F.one_hot(torch.tensor([2]), num_classes=4)
loss_func = CrossEntropyLoss()
loss = loss_func(predictions.float(), ground_truth.float())
print(f"loss: {loss}")

# do the backpropagation here
loss.backward()
print(f"After backpropagation: ")
# the weights and biases of all the layers essentially get updated based on the gradient that was calculated    
print(f"layer 1 gradients: {model[0].weight.grad, model[0].bias.grad}")
print(f"layer 2 gradients: {model[1].weight.grad, model[1].bias.grad}")
print(f"layer 3 gradients: {model[2].weight.grad, model[2].bias.grad}")