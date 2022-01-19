import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim, name, chkpt_dir):
        super(FeedForwardNN, self).__init__()
    
        self.chkpt_file = os.path.join(chkpt_dir, name)
    
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
    
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    
        self.to(self.device)
    
    def forward(self, obs):
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        output = self.layer3(x)
        return output

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
