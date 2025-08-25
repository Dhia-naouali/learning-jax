from flax import linen as nn



class NN(nn.Module):
    hidden_dim: int
    num_classes: int
    def setup(self):
        self.fc1 = nn.Dense(self.hidden_dim)
        self.fc2 = nn.Dense(self.hidden_dim)
        self.fc3 = nn.Dense(self.num_classes)
        
    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = nn.relu(self.fc3(x))
        return x
    
    