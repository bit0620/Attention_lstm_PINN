import torch
import torch.nn as nn

class FFN(nn.Module):
    """

    Feedforward neural network with one hidden layer.
    """


    def __init__(self, in_dim, hidden_dim, out_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x

in_dim = 10
hidden_dim = 20
out_dim = 1

ffn = FFN(in_dim, hidden_dim, out_dim)

x = torch.randn(1, in_dim)
print(ffn(x))

print(ffn(x).shape)

print(ffn(x).dtype)
print(ffn(x).device)
print(ffn(x).requires_grad)


