import torch.nn as nn
import torch

class BatchNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine:bool = True, track_running_stats:bool = True):
        super().__init__()
        
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
    
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.channels))
            self.beta = nn.Parameter(torch.zeros(self.channels))
    
        if self.track_running_stats:
            self.register_buffer("exp_mean", torch.zeros(self.channels))
            self.register_buffer("exp_var", torch.ones(self.channels))
    
    def forward(self, x: torch.Tensor):
        x_shape = x.shape 
        batch_size = x_shape[0]

        assert self.channels == x_shape[1]

        x = x.view(batch_size, self.channels, -1)

        if self.track_running_stats:
            mean = x.mean(dim=[0,2])
            mean_x2 = (x ** 2).mean(dim=[0,2])
            var = mean_x2 - mean ** 2

            self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
            self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var
        
        else:
            mean = self.exp_mean
            var = self.exp_var
        
        x_hat = (x - mean.view(1, -1, 1))/torch.sqrt(var + self.eps).view(1, -1, 1)

        if self.affine:
            x_hat = self.gamma.view(1, -1, 1) * x_hat + self.beta.view(1, -1, 1)
        
        return x_hat.view(x_shape)

if __name__ == '__main__':
    x = torch.randn([2, 3, 2, 4])
    print(f"Original : {x}")
    BN = BatchNorm(3)
    print(f"Mean : {BN.exp_mean}, Var : {BN.exp_var}")
    y = BN(x)
    print(f"Normalized : {y}")
    print(f"Mean : {BN.exp_mean}, Var : {BN.exp_var}")
