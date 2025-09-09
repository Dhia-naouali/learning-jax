from flax import linen as nn

class CNN(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.relu(
            nn.Conv(32, kernel_size=(3,3))(x)            
        )
        x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2))
        
        x = nn.relu(
            nn.Conv(64, kernel_size=(3,3))(x)
        )
        x = nn.avg_pool(x, (2,2), (2,2))
        
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x