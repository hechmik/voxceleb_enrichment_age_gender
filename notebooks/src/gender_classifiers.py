import torch
from torch.autograd import Variable
from torch.nn import functional as F

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

class FC2(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout, dropout_p=0.5):
        super(FC2, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, output_dim)
        self.apply_dropout = dropout
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        if self.apply_dropout:
            x = self.dropout(x)
        outputs = self.fc2(x)
        return outputs
    
    
class FC4(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout, dropout_p=0.5):
        super(FC4, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.fc4 = torch.nn.Linear(512, output_dim)
        self.apply_dropout = dropout
        self.do1 = torch.nn.Dropout(dropout_p)
        self.do2 = torch.nn.Dropout(dropout_p)
        self.do3 = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        if self.apply_dropout:
            x = self.do1(x)
        x = self.fc2(x)
        x = self.relu(x)
        if self.apply_dropout:
            x = self.do2(x)
        x = self.fc3(x)
        x = self.relu(x)
        if self.apply_dropout:
            x = self.do3(x)
        outputs = self.fc4(x)
        return outputs