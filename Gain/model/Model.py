import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, dim):
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(dim * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, dim)
        
    def forward(self, x, m):
        inputs = torch.cat(dim=1, tensors=[x, m])
        
        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        
        return torch.sigmoid(self.fc3(out)) # 0~1 사이 값 출력

class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(dim * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, dim)
        
    def forward(self, x, h):
        inputs = torch.cat(dim=1, tensors=[x, h])
        
        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        
        return torch.sigmoid(self.fc3(out)) # 0~1 사이 확률 출력